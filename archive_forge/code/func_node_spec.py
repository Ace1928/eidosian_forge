import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
@property
def node_spec(self) -> '_NodeSpec':
    if self._node_spec is not None:
        return self._node_spec
    raise InvalidOperationError(f'node_spec is not set for {self}')