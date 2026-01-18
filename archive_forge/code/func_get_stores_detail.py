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
def get_stores_detail(self, req):
    enabled_backends = CONF.enabled_backends
    stores = self.get_stores(req).get('stores')
    try:
        api_policy.DiscoveryAPIPolicy(req.context, enforcer=self.policy).stores_info_detail()
        store_mapper = {'rbd': self._get_rbd_properties, 'file': self._get_file_properties, 'cinder': self._get_cinder_properties, 'swift': self._get_swift_properties, 's3': self._get_s3_properties, 'http': self._get_http_properties}
        for store in stores:
            store_type = enabled_backends[store['id']]
            store['type'] = store_type
            store_detail = g_store.get_store_from_store_identifier(store['id'])
            store['properties'] = store_mapper.get(store_type)(store_detail)
            store['weight'] = getattr(CONF, store['id']).weight
    except exception.Forbidden as e:
        LOG.debug('User not permitted to view details')
        raise webob.exc.HTTPForbidden(explanation=e.msg)
    return {'stores': stores}