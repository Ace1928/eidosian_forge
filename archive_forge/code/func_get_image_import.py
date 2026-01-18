import copy
import glance_store as g_store
from oslo_config import cfg
from oslo_log import log as logging
import oslo_serialization.jsonutils as json
import webob.exc
from glance.api import policy
from glance.api.v2 import policy as api_policy
from glance.common import exception
from glance.common import wsgi
import glance.db
from glance.i18n import _
from glance.quota import keystone as ks_quota
def get_image_import(self, req):
    import_methods = {'description': 'Import methods available.', 'type': 'array', 'value': CONF.get('enabled_import_methods')}
    return {'import-methods': import_methods}