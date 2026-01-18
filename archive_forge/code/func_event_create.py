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
def event_create(context, values):
    if 'stack_id' in values and cfg.CONF.max_events_per_stack:
        check = 2.0 / cfg.CONF.event_purge_batch_size > random.uniform(0, 1)
        if check and _event_count_all_by_stack(context, values['stack_id']) >= cfg.CONF.max_events_per_stack:
            try:
                _delete_event_rows(context, values['stack_id'], cfg.CONF.event_purge_batch_size)
            except db_exception.DBError as exc:
                LOG.error('Failed to purge events: %s', str(exc))
    event_ref = models.Event()
    event_ref.update(values)
    event_ref.save(context.session)
    result = context.session.query(models.Event).filter_by(id=event_ref.id).options(orm.joinedload(models.Event.rsrc_prop_data)).first()
    return result