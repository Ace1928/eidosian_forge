import collections
import copy
from oslo_context import context as oslo_context
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.db import api as db_api
from neutron_lib.policy import _engine as policy_engine
@property
def is_advsvc(self):
    LOG.warning("Method 'is_advsvc' is deprecated since 2023.2 release (neutron-lib 3.8.0) and will be removed once support for the old RBAC API policies will be removed from Neutron. Please use method 'is_service_role' instead.")
    return self.is_service_role