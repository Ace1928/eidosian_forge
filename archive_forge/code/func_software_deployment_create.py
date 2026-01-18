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
def software_deployment_create(context, values):
    obj_ref = models.SoftwareDeployment()
    obj_ref.update(values)
    try:
        obj_ref.save(context.session)
    except db_exception.DBReferenceError:
        err_msg = _('Config with id %s not found') % values['config_id']
        raise exception.Invalid(reason=err_msg)
    return _software_deployment_get(context, obj_ref.id)