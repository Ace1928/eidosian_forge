import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Optional, Tuple, Type, Union
from typing_extensions import Literal, Protocol
from .typing import AnyArgTCallable, AnyCallable
from .utils import GetterDict
from .version import compiled
def inherit_config(self_config: 'ConfigType', parent_config: 'ConfigType', **namespace: Any) -> 'ConfigType':
    if not self_config:
        base_classes: Tuple['ConfigType', ...] = (parent_config,)
    elif self_config == parent_config:
        base_classes = (self_config,)
    else:
        base_classes = (self_config, parent_config)
    namespace['json_encoders'] = {**getattr(parent_config, 'json_encoders', {}), **getattr(self_config, 'json_encoders', {}), **namespace.get('json_encoders', {})}
    return type('Config', base_classes, namespace)