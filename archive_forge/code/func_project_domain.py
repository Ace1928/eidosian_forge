from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def project_domain(self):
    if not self.__project_domain:
        if self.project and self.project.get('domain_id'):
            self.__project_domain = PROVIDERS.resource_api.get_domain(self.project['domain_id'])
    return self.__project_domain