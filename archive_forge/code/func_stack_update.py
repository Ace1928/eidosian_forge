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
@retry_on_db_error
@context_manager.writer
def stack_update(context, stack_id, values, exp_trvsl=None):
    query = context.session.query(models.Stack).filter(and_(models.Stack.id == stack_id), models.Stack.deleted_at.is_(None))
    if not context.is_admin:
        query = query.filter(sqlalchemy.or_(models.Stack.tenant == context.tenant_id, models.Stack.stack_user_project_id == context.tenant_id))
    if exp_trvsl is not None:
        query = query.filter(models.Stack.current_traversal == exp_trvsl)
    rows_updated = query.update(values, synchronize_session=False)
    if not rows_updated:
        LOG.debug('Did not set stack state with values %(vals)s, stack id: %(id)s with expected traversal: %(trav)s', {'id': stack_id, 'vals': str(values), 'trav': str(exp_trvsl)})
        if not _stack_get(context, stack_id, eager_load=False):
            raise exception.NotFound(_('Attempt to update a stack with id: %(id)s %(msg)s') % {'id': stack_id, 'msg': 'that does not exist'})
    return rows_updated is not None and rows_updated > 0