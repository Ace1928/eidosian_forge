import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def normalize_user(self, user, default_mapping_domain):
    super(RuleProcessorToHonorDomainOption, self).normalize_user(user, default_mapping_domain)
    if not user.get('domain'):
        LOG.debug('Configuring the domain [%s] for user [%s].', default_mapping_domain, user)
        user['domain'] = default_mapping_domain
    else:
        LOG.debug('The user [%s] was configured with a domain. Therefore, we do not need to define.', user)