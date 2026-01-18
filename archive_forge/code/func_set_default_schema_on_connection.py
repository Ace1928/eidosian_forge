from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
@register.init
def set_default_schema_on_connection(cfg, dbapi_connection, schema_name):
    raise NotImplementedError('backend does not implement a schema name set function: %s' % (cfg.db.url,))