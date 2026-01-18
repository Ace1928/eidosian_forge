from oslo_config import cfg
from keystone.conf import constants
from keystone.conf import utils
def setup_authentication(conf=None):
    """Register non-default auth methods (used by extensions, etc)."""
    if conf is None:
        conf = cfg.CONF
    for method_name in conf.auth.methods:
        if method_name not in constants._DEFAULT_AUTH_METHODS:
            option = cfg.StrOpt(method_name)
            _register_auth_plugin_opt(conf, option)