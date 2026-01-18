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
def recursive_sort(obj):
    """Recursively sort list in iterables for comparison."""
    if isinstance(obj, dict):
        for v in obj.values():
            recursive_sort(v)
    elif isinstance(obj, list):
        obj.sort()
        for i in obj:
            recursive_sort(i)
    return obj