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
def get_stores(self, req):
    enabled_backends = CONF.enabled_backends
    if not enabled_backends:
        msg = _('Multi backend is not supported at this site.')
        raise webob.exc.HTTPNotFound(explanation=msg)
    backends = []
    for backend in enabled_backends:
        if backend.startswith('os_glance_'):
            continue
        stores = {}
        stores['id'] = backend
        description = getattr(CONF, backend).store_description
        if description:
            stores['description'] = description
        if backend == CONF.glance_store.default_backend:
            stores['default'] = 'true'
        if enabled_backends[backend] == 'http':
            stores['read-only'] = 'true'
        backends.append(stores)
    return {'stores': backends}