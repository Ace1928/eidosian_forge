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
def assert_enabled_identity_provider(federation_api, idp_id):
    identity_provider = federation_api.get_idp(idp_id)
    if identity_provider.get('enabled') is not True:
        msg = 'Identity Provider %(idp)s is disabled' % {'idp': idp_id}
        tr_msg = _('Identity Provider %(idp)s is disabled') % {'idp': idp_id}
        LOG.debug(msg)
        raise exception.Forbidden(tr_msg)