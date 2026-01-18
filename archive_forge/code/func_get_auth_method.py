from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def get_auth_method(method_name):
    global AUTH_METHODS
    if method_name not in AUTH_METHODS:
        raise exception.AuthMethodNotSupported()
    return AUTH_METHODS[method_name]