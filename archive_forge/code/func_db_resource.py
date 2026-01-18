from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def db_resource(current_template_id, created_at=None, updated_at=None):
    db_res = resource_objects.Resource(stack.context)
    db_res['id'] = current_template_id
    db_res['name'] = 'A'
    db_res['current_template_id'] = current_template_id
    db_res['action'] = 'UPDATE' if updated_at else 'CREATE'
    db_res['status'] = 'COMPLETE'
    db_res['updated_at'] = updated_at
    db_res['created_at'] = created_at
    db_res['replaced_by'] = None
    return db_res