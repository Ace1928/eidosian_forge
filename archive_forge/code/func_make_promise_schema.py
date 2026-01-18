import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
@classmethod
def make_promise_schema(cls, obj: Dict[str, Any], *, resolve: bool=True) -> Type[BaseModel]:
    """Create a schema for a promise dict (referencing a registry function)
        by inspecting the function signature.
        """
    reg_name, func_name = cls.get_constructor(obj)
    if not resolve and (not cls.has(reg_name, func_name)):
        return EmptySchema
    func = cls.get(reg_name, func_name)
    id_keys = [k for k in obj.keys() if k.startswith('@')]
    sig_args: Dict[str, Any] = {id_keys[0]: (str, ...)}
    for param in inspect.signature(func).parameters.values():
        annotation = param.annotation if param.annotation != param.empty else Any
        default = param.default if param.default != param.empty else ...
        if param.kind == param.VAR_POSITIONAL:
            spread_annot = Sequence[annotation]
            sig_args[ARGS_FIELD_ALIAS] = (spread_annot, default)
        else:
            name = RESERVED_FIELDS.get(param.name, param.name)
            sig_args[name] = (annotation, default)
    sig_args['__config__'] = _PromiseSchemaConfig
    return create_model('ArgModel', **sig_args)