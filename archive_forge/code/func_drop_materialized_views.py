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
def drop_materialized_views(cfg, eng):
    inspector = inspect(eng)
    mview_names = inspector.get_materialized_view_names()
    with eng.begin() as conn:
        for vname in mview_names:
            conn.exec_driver_sql(f'DROP MATERIALIZED VIEW {vname}')
    if config.requirements.schemas.enabled_for_config(cfg):
        mview_names = inspector.get_materialized_view_names(schema=cfg.test_schema)
        with eng.begin() as conn:
            for vname in mview_names:
                conn.exec_driver_sql(f'DROP MATERIALIZED VIEW {cfg.test_schema}.{vname}')