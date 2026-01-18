import datetime
import functools
import itertools
import random
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import orm
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import filters as db_filters
from heat.db import models
from heat.db import utils as db_utils
from heat.engine import environment as heat_environment
from heat.rpc import api as rpc_api
@context_manager.writer
def stack_create(context, values):
    stack_ref = models.Stack()
    stack_ref.update(values)
    stack_name = stack_ref.name
    stack_ref.save(context.session)
    earliest = _stack_get_by_name(context, stack_name)
    if earliest is not None and earliest.id != stack_ref.id:
        context.session.query(models.Stack).filter_by(id=stack_ref.id).delete()
        raise exception.StackExists(stack_name=stack_name)
    return stack_ref