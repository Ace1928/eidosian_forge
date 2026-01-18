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
def filter_limit_query(model, query, hints):
    """Apply filtering and limit to a query.

    :param model: table model
    :param query: query to apply filters to
    :param hints: contains the list of filters and limit details.  This may
                  be None, indicating that there are no filters or limits
                  to be applied. If it's not None, then any filters
                  satisfied here will be removed so that the caller will
                  know if any filters remain.

    :returns: query updated with any filters and limits satisfied

    """
    if hints is None:
        return query
    query = _filter(model, query, hints)
    if hints.cannot_match:
        return []
    if not hints.filters:
        return _limit(query, hints)
    else:
        return query