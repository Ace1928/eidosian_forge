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
def get_temp_table_name(cfg, eng, base_name):
    """Specify table name for creating a temporary Table.

    Dialect-specific implementations of this method will return the
    name to use when creating a temporary table for testing,
    e.g., in the define_temp_tables method of the
    ComponentReflectionTest class in suite/test_reflection.py

    Default to just the base name since that's what most dialects will
    use. The mssql dialect's implementation will need a "#" prepended.
    """
    return base_name