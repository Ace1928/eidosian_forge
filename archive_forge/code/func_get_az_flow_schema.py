from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger
from lazyops.utils.lazy import lazy_import
from lazyops.utils.helpers import fail_after
from typing import Any, Callable, Dict, List, Optional, Union, Type
def get_az_flow_schema(name: str) -> 'AZFlowSchema':
    """
    Returns the AZFlowSchema
    """
    global _az_flow_schemas
    if name not in _az_flow_schemas:
        raise ValueError(f'Invalid AuthZero Flow: {name}, must be one of {list(_az_flow_schemas.keys())}')
    if isinstance(_az_flow_schemas[name], str):
        _az_flow_schemas[name] = lazy_import(_az_flow_schemas[name])
    return _az_flow_schemas[name]