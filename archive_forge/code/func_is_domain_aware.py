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
def is_domain_aware(self):
    return False