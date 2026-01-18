import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def ldap2py(val):
    """Convert an LDAP formatted value to Python type used by OpenStack.

    Virtually all LDAP values are stored as UTF-8 encoded strings.
    OpenStack prefers values which are unicode friendly.

    :param val: LDAP formatted value
    :returns: val converted to preferred Python type
    """
    return utf8_decode(val)