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
def stack_tags_set(context, stack_id, tags):
    _stack_tags_delete(context, stack_id)
    result = []
    for tag in tags:
        stack_tag = models.StackTag()
        stack_tag.tag = tag
        stack_tag.stack_id = stack_id
        stack_tag.save(session=context.session)
        result.append(stack_tag)
    return result or None