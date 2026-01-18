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
def affirm_unique(self, values):
    if values.get('name') is not None:
        try:
            self.get_by_name(values['name'])
        except exception.NotFound:
            pass
        else:
            raise exception.Conflict(type=self.options_name, details=_('Duplicate name, %s.') % values['name'])
    if values.get('id') is not None:
        try:
            self.get(values['id'])
        except exception.NotFound:
            pass
        else:
            raise exception.Conflict(type=self.options_name, details=_('Duplicate ID, %s.') % values['id'])