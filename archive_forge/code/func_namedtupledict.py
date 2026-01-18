from collections import namedtuple
from typing import Any, Dict, NewType, Optional, Sequence, Tuple, Type
import numpy
import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import IR_VERSION, ModelProto, NodeProto
def namedtupledict(typename: str, field_names: Sequence[str], *args: Any, **kwargs: Any) -> Type[Tuple[Any, ...]]:
    field_names_map = {n: i for i, n in enumerate(field_names)}
    kwargs.setdefault('rename', True)
    data = namedtuple(typename, field_names, *args, **kwargs)

    def getitem(self: Any, key: Any) -> Any:
        if isinstance(key, str):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)
    data.__getitem__ = getitem
    return data