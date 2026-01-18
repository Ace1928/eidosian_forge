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
def get_method_data(self, method):
    """Get the auth method payload.

        :returns: auth method payload

        """
    if method not in self.auth['identity']['methods']:
        raise exception.ValidationError(attribute=method, target='identity')
    return self.auth['identity'][method]