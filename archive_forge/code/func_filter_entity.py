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
def filter_entity(entity_ref):
    """Filter out private items in an entity dict.

    :param entity_ref:  the entity dictionary. The 'dn' field will be removed.
        'dn' is used in LDAP, but should not be returned to the user.  This
        value may be modified.

    :returns: entity_ref

    """
    if entity_ref:
        entity_ref.pop('dn', None)
    return entity_ref