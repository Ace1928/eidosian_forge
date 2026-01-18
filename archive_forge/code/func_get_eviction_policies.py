import dill
from pydantic import BaseModel, Field, root_validator
from pydantic.types import ByteSize
from typing import Union, Any, Dict, Optional
from lazyops.configs.base import DefaultSettings
from lazyops.types import lazyproperty
from lazyops.libs.sqlcache.constants import DEFAULT_SETTINGS, OPTIMIZED_SETTINGS, DBNAME
def get_eviction_policies(table_name: str):
    return {'none': {'init': None, 'get': None, 'cull': None}, 'least-recently-stored': {'init': f'CREATE INDEX IF NOT EXISTS {table_name}_store_time ON {table_name} (store_time)', 'get': None, 'cull': 'SELECT {fields} FROM ' + table_name + ' ORDER BY store_time LIMIT ?'}, 'least-recently-used': {'init': f'CREATE INDEX IF NOT EXISTS {table_name}_access_time ON {table_name} (access_time)', 'get': 'access_time = {now}', 'cull': 'SELECT {fields} FROM ' + table_name + ' ORDER BY access_time LIMIT ?'}, 'least-frequently-used': {'init': f'CREATE INDEX IF NOT EXISTS {table_name}_access_count ON {table_name} (access_count)', 'get': 'access_count = access_count + 1', 'cull': 'SELECT {fields} FROM ' + table_name + ' ORDER BY access_count LIMIT ?'}}