import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_attribute_ref(name: str, attr_type: AttributeProto.AttributeType, doc_string: Optional[str]=None) -> AttributeProto:
    """Make an AttributeProto holding a reference to the parent function's attribute of given name and type."""
    attr = AttributeProto()
    attr.name = name
    attr.type = attr_type
    if doc_string:
        attr.doc_string = doc_string
    return attr