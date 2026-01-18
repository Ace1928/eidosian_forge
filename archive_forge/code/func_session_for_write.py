import datetime
import functools
import pytz
from oslo_db import exception as db_exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import models
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from osprofiler import opts as profiler
import osprofiler.sqlalchemy
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.orm.attributes import flag_modified, InstrumentedAttribute
from sqlalchemy import types as sql_types
from keystone.common import driver_hints
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def session_for_write():
    if _TESTING_USE_GLOBAL_CONTEXT_MANAGER:
        writer = enginefacade.writer
    else:
        writer = _get_main_context_manager().writer
    return _wrap_session(writer.using(_get_context()))