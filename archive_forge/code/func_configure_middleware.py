import uuid
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystonemiddleware.auth_token import _base
from keystonemiddleware.tests.unit.auth_token import base
def configure_middleware(self, auth_type, **kwargs):
    opts = loading.get_auth_plugin_conf_options(auth_type)
    self.cfg.register_opts(opts, group=_base.AUTHTOKEN_GROUP)
    loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
    self.cfg.config(group=_base.AUTHTOKEN_GROUP, auth_type=auth_type, **kwargs)