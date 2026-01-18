import uuid
import ldap.filter
from oslo_log import log
from oslo_log import versionutils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends.ldap import models
def mask_enabled_attribute(self, values):
    value = values['enabled']
    values.setdefault('enabled_nomask', int(self.enabled_default))
    if value != (values['enabled_nomask'] & self.enabled_mask != self.enabled_mask):
        values['enabled_nomask'] ^= self.enabled_mask
    values['enabled'] = values['enabled_nomask']
    del values['enabled_nomask']