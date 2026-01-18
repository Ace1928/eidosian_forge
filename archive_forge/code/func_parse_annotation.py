import copy
import inspect
import re
from typing import (
from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict
@classmethod
def parse_annotation(cls, annotation: Any, param: Optional[inspect.Parameter]=None, none_as_other: bool=True) -> 'AnnotatedParam':
    if annotation == type(None):
        return OtherParam(param) if none_as_other else NoneParam(param)
    if annotation == inspect.Parameter.empty:
        if param is not None and param.kind == param.VAR_POSITIONAL:
            return PositionalParam(param)
        if param is not None and param.kind == param.VAR_KEYWORD:
            return KeywordParam(param)
        return OtherParam(param) if none_as_other else NoneParam(param)
    load_entry_point(cls._ENTRYPOINT)
    for tp, _, _, matcher in cls._REGISTERED:
        if matcher(annotation):
            return tp(param)
    if param is not None and param.kind == param.VAR_POSITIONAL:
        return PositionalParam(param)
    if param is not None and param.kind == param.VAR_KEYWORD:
        return KeywordParam(param)
    return OtherParam(param)