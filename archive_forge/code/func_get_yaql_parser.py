import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
@classmethod
def get_yaql_parser(cls):
    if cls._parser is None:
        global_options = {'yaql.limitIterators': cfg.CONF.yaql.limit_iterators, 'yaql.memoryQuota': cfg.CONF.yaql.memory_quota}
        cls._parser = yaql.YaqlFactory().create(global_options)
        cls._context = yaql.create_context()
    return cls._parser