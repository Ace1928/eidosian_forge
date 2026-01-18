import collections
import math
import numbers
from typing import Any, Dict as PythonDict, Hashable, List as PythonList, Optional, Sequence, Tuple as PythonTuple, Type
import weakref
from tensorflow.core.function.trace_type import default_types_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.core.function.trace_type import util
from tensorflow.python.types import trace
@classmethod
def from_type_and_attributes(cls, attrs_type: Any, attributes: PythonTuple[trace.TraceType]) -> 'Attrs':
    return Attrs(attrs_type.__name__, tuple((attr.name for attr in attrs_type.__attrs_attrs__)), attributes, attrs_type)