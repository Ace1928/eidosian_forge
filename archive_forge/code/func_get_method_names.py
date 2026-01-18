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
def get_method_names(self):
    """Return the identity method names.

        :returns: list of auth method names

        """
    method_names = []
    for method in self.auth['identity']['methods']:
        if method not in method_names:
            method_names.append(method)
    return method_names