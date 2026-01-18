import collections
import copy
from oslo_context import context as oslo_context
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import timeutils
from neutron_lib.db import api as db_api
from neutron_lib.policy import _engine as policy_engine
def get_admin_context_without_session():
    return ContextBase(user_id=None, tenant_id=None, is_admin=True).elevated()