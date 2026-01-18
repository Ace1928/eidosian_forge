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
def get_all_filtered(self, hints, query=None):
    if self.ldap_filter:
        query = (query or '') + self.ldap_filter
    query = self.filter_query(hints, query)
    return [common_ldap.filter_entity(group) for group in self.get_all(query, hints)]