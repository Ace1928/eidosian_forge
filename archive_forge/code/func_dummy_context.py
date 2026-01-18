import random
import string
import uuid
import warnings
import fixtures
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
import sqlalchemy
from sqlalchemy import exc as sqla_exc
from heat.common import context
from heat.db import api as db_api
from heat.db import models
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine import stack
from heat.engine import template
def dummy_context(user='test_username', tenant_id='test_tenant_id', password='', roles=None, user_id=None, trust_id=None, region_name=None, is_admin=False):
    roles = roles or []
    return context.RequestContext.from_dict({'tenant_id': tenant_id, 'tenant': 'test_tenant', 'username': user, 'user_id': user_id, 'password': password, 'roles': roles, 'is_admin': is_admin, 'auth_url': 'http://server.test:5000/v2.0', 'auth_token': 'abcd1234', 'trust_id': trust_id, 'region_name': region_name})