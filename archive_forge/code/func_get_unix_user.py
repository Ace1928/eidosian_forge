import base64
import collections.abc
import contextlib
import grp
import hashlib
import itertools
import os
import pwd
import uuid
from cryptography import x509
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import urllib
from keystone.common import password_hashing
import keystone.conf
from keystone import exception
from keystone.i18n import _
def get_unix_user(user=None):
    """Get the uid and user name.

    This is a convenience utility which accepts a variety of input
    which might represent a unix user. If successful it returns the uid
    and name. Valid input is:

    string
        A string is first considered to be a user name and a lookup is
        attempted under that name. If no name is found then an attempt
        is made to convert the string to an integer and perform a
        lookup as a uid.

    int
        An integer is interpreted as a uid.

    None
        None is interpreted to mean use the current process's
        effective user.

    If the input is a valid type but no user is found a KeyError is
    raised. If the input is not a valid type a TypeError is raised.

    :param object user: string, int or None specifying the user to
                        lookup.

    :returns: tuple of (uid, name)

    """
    if isinstance(user, str):
        try:
            user_info = pwd.getpwnam(user)
        except KeyError:
            try:
                i = int(user)
            except ValueError:
                raise KeyError("user name '%s' not found" % user)
            try:
                user_info = pwd.getpwuid(i)
            except KeyError:
                raise KeyError('user id %d not found' % i)
    elif isinstance(user, int):
        try:
            user_info = pwd.getpwuid(user)
        except KeyError:
            raise KeyError('user id %d not found' % user)
    elif user is None:
        user_info = pwd.getpwuid(os.geteuid())
    else:
        user_cls_name = reflection.get_class_name(user, fully_qualified=False)
        raise TypeError('user must be string, int or None; not %s (%r)' % (user_cls_name, user))
    return (user_info.pw_uid, user_info.pw_name)