import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
@contextlib.contextmanager
def record_sql_queries(hub, cursor, query, params_list, paramstyle, executemany, record_cursor_repr=False):
    if hub.client and hub.client.options['_experiments'].get('record_sql_params', False):
        if not params_list or params_list == [None]:
            params_list = None
        if paramstyle == 'pyformat':
            paramstyle = 'format'
    else:
        params_list = None
        paramstyle = None
    query = _format_sql(cursor, query)
    data = {}
    if params_list is not None:
        data['db.params'] = params_list
    if paramstyle is not None:
        data['db.paramstyle'] = paramstyle
    if executemany:
        data['db.executemany'] = True
    if record_cursor_repr and cursor is not None:
        data['db.cursor'] = cursor
    with capture_internal_exceptions():
        hub.add_breadcrumb(message=query, category='query', data=data)
    with hub.start_span(op=OP.DB, description=query) as span:
        for k, v in data.items():
            span.set_data(k, v)
        yield span