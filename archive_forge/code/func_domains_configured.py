import copy
import functools
import itertools
import operator
import os
import threading
import uuid
from oslo_config import cfg
from oslo_log import log
from pycadf import reason
from keystone import assignment  # TODO(lbragstad): Decouple this dependency
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping
from keystone import notifications
from oslo_utils import timeutils
def domains_configured(f):
    """Wrap API calls to lazy load domain configs after init.

    This is required since the assignment manager needs to be initialized
    before this manager, and yet this manager's init wants to be
    able to make assignment calls (to build the domain configs).  So
    instead, we check if the domains have been initialized on entry
    to each call, and if requires load them,

    """

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.domain_configs.configured and CONF.identity.domain_specific_drivers_enabled:
            with self.domain_configs.lock:
                if not self.domain_configs.configured:
                    self.domain_configs.setup_domain_drivers(self.driver, PROVIDERS.resource_api)
        return f(self, *args, **kwargs)
    return wrapper