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
def is_rdn_equal(rdn1, rdn2):
    """Return True if and only if the RDNs are equal.

    * RDNs must have the same number of AVAs.
    * Each AVA of the RDNs must be the equal for the same attribute type. The
      order isn't significant. Note that an attribute type will only be in one
      AVA in an RDN, otherwise the DN wouldn't be valid.
    * Attribute types aren't case sensitive. Note that attribute type
      comparison is more complicated than implemented. This function only
      compares case-insentive. The code should handle multiple names for an
      attribute type (e.g., cn, commonName, and 2.5.4.3 are the same).

    Note that this function uses is_ava_value_equal to compare AVAs so the
    limitations of that function apply here.

    """
    if len(rdn1) != len(rdn2):
        return False
    for attr_type_1, val1, dummy in rdn1:
        found = False
        for attr_type_2, val2, dummy in rdn2:
            if attr_type_1.lower() != attr_type_2.lower():
                continue
            found = True
            if not is_ava_value_equal(attr_type_1, val1, val2):
                return False
            break
        if not found:
            return False
    return True