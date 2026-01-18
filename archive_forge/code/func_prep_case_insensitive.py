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
def prep_case_insensitive(value):
    """Prepare a string for case-insensitive comparison.

    This is defined in RFC4518. For simplicity, all this function does is
    lowercase all the characters, strip leading and trailing whitespace,
    and compress sequences of spaces to a single space.
    """
    value = re.sub('\\s+', ' ', value.strip().lower())
    return value