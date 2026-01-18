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
def is_ava_value_equal(attribute_type, val1, val2):
    """Return True if and only if the AVAs are equal.

    When comparing AVAs, the equality matching rule for the attribute type
    should be taken into consideration. For simplicity, this implementation
    does a case-insensitive comparison.

    Note that this function uses prep_case_insenstive so the limitations of
    that function apply here.

    """
    return prep_case_insensitive(val1) == prep_case_insensitive(val2)