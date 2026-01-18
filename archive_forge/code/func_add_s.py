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
def add_s(self, dn, modlist):
    ldap_attrs = [(kind, [py2ldap(x) for x in safe_iter(values)]) for kind, values in modlist]
    logging_attrs = [(kind, values if kind != 'userPassword' else ['****']) for kind, values in ldap_attrs]
    LOG.debug('LDAP add: dn=%s attrs=%s', dn, logging_attrs)
    ldap_attrs_utf8 = [(kind, [utf8_encode(x) for x in safe_iter(values)]) for kind, values in ldap_attrs]
    return self.conn.add_s(dn, ldap_attrs_utf8)