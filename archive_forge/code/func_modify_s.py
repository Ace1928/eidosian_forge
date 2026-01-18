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
def modify_s(self, dn, modlist):
    ldap_modlist = [(op, kind, None if values is None else [py2ldap(x) for x in safe_iter(values)]) for op, kind, values in modlist]
    logging_modlist = [(op, kind, values if kind != 'userPassword' else ['****']) for op, kind, values in ldap_modlist]
    LOG.debug('LDAP modify: dn=%s modlist=%s', dn, logging_modlist)
    ldap_modlist_utf8 = [(op, kind, None if values is None else [utf8_encode(x) for x in safe_iter(values)]) for op, kind, values in ldap_modlist]
    return self.conn.modify_s(dn, ldap_modlist_utf8)