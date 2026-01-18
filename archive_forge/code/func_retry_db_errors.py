import contextlib
import copy
import functools
import weakref
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import excutils
from osprofiler import opts as profiler_opts
import osprofiler.sqlalchemy
from pecan import util as p_util
import sqlalchemy
from sqlalchemy import event  # noqa
from sqlalchemy import exc as sql_exc
from sqlalchemy import orm
from sqlalchemy.orm import exc
from neutron_lib._i18n import _
from neutron_lib.db import model_base
from neutron_lib import exceptions
from neutron_lib.objects import exceptions as obj_exc
def retry_db_errors(f):
    """Nesting-safe retry decorator with auto-arg-copy and logging.

    Retry decorator for all functions which do not accept a context as an
    argument. If the function accepts a context, use
    'retry_if_session_inactive' below.

    If retriable errors are retried and exceed the count, they will be tagged
    with a flag so is_retriable will no longer recognize them as retriable.
    This prevents multiple applications of this decorator (and/or the one
    below) from retrying the same exception.
    """

    @_tag_retriables_as_unretriable
    @_retry_db_errors
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        context_reference = None
        try:
            context_reference = kwargs.pop('_context_reference', None)
            dup_args = [_copy_if_lds(a) for a in args]
            dup_kwargs = {k: _copy_if_lds(v) for k, v in kwargs.items()}
            return f(*dup_args, **dup_kwargs)
        except Exception as e:
            with excutils.save_and_reraise_exception():
                if is_retriable(e):
                    LOG.debug('Retry wrapper got retriable exception: %s', e)
                    if context_reference and context_reference.session:
                        context_reference.session.rollback()
    return wrapped