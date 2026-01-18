import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_operatorsetid(domain: str, version: int) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Args:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    Returns:
        OperatorSetIdProto
    """
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetid