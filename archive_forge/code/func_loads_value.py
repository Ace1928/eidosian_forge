from __future__ import annotations
import collections.abc
import struct
import uuid
import numpy as np
import symengine
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.circuit import CASE_DEFAULT, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.qpy import common, formats, exceptions, type_keys
def loads_value(type_key, binary_data, version, vectors, *, clbits=(), cregs=None, use_symengine=False):
    """Deserialize input binary data to value object.

    Args:
        type_key (ValueTypeKey): Type enum information.
        binary_data (bytes): Data to deserialize.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.
        clbits (Sequence[Clbit]): Clbits in the current scope.
        cregs (Mapping[str, ClassicalRegister]): Classical registers in the current scope.
        use_symengine (bool): If True, symbolic objects will be de-serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.

    Returns:
        any: Deserialized value object.

    Raises:
        QpyError: Serializer for given format is not ready.
    """
    if isinstance(type_key, bytes):
        type_key = type_keys.Value(type_key)
    if type_key == type_keys.Value.INTEGER:
        return struct.unpack('!q', binary_data)[0]
    if type_key == type_keys.Value.FLOAT:
        return struct.unpack('!d', binary_data)[0]
    if type_key == type_keys.Value.COMPLEX:
        return complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
    if type_key == type_keys.Value.NUMPY_OBJ:
        return common.data_from_binary(binary_data, np.load)
    if type_key == type_keys.Value.STRING:
        return binary_data.decode(common.ENCODE)
    if type_key == type_keys.Value.NULL:
        return None
    if type_key == type_keys.Value.CASE_DEFAULT:
        return CASE_DEFAULT
    if type_key == type_keys.Value.PARAMETER_VECTOR:
        return common.data_from_binary(binary_data, _read_parameter_vec, vectors=vectors)
    if type_key == type_keys.Value.PARAMETER:
        return common.data_from_binary(binary_data, _read_parameter)
    if type_key == type_keys.Value.PARAMETER_EXPRESSION:
        if version < 3:
            return common.data_from_binary(binary_data, _read_parameter_expression)
        else:
            return common.data_from_binary(binary_data, _read_parameter_expression_v3, vectors=vectors, use_symengine=use_symengine)
    if type_key == type_keys.Value.EXPRESSION:
        return common.data_from_binary(binary_data, _read_expr, clbits=clbits, cregs=cregs or {})
    raise exceptions.QpyError(f'Serialization for {type_key} is not implemented in value I/O.')