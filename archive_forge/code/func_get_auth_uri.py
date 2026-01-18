from keystoneauth1 import discover as ks_discover
from keystoneauth1 import session as ks_session
from oslo_config import cfg
from oslo_utils import importutils
from heat.common import config
def get_auth_uri(v3=True):
    if cfg.CONF.clients_keystone.auth_uri:
        session = ks_session.Session(**config.get_ssl_options('keystone'))
        discover = ks_discover.Discover(session=session, url=cfg.CONF.clients_keystone.auth_uri)
        return discover.url_for('3.0')
    else:
        importutils.import_module('keystonemiddleware.auth_token')
        try:
            auth_uri = cfg.CONF.keystone_authtoken.www_authenticate_uri
        except cfg.NoSuchOptError:
            auth_uri = cfg.CONF.keystone_authtoken.auth_uri
        return auth_uri.replace('v2.0', 'v3') if auth_uri and v3 else auth_uri