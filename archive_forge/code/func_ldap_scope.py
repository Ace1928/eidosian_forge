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
def ldap_scope(scope):
    try:
        return LDAP_SCOPES[scope]
    except KeyError:
        raise ValueError(_('Invalid LDAP scope: %(scope)s. Choose one of: %(options)s') % {'scope': scope, 'options': ', '.join(LDAP_SCOPES.keys())})