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
def dumps_value(obj, *, index_map=None, use_symengine=False):
    """Serialize input value object.

    Args:
        obj (any): Arbitrary value object to serialize.
        index_map (dict): Dictionary with two keys, "q" and "c".  Each key has a value that is a
            dictionary mapping :class:`.Qubit` or :class:`.Clbit` instances (respectively) to their
            integer indices.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.

    Returns:
        tuple: TypeKey and binary data.

    Raises:
        QpyError: Serializer for given format is not ready.
    """
    type_key = type_keys.Value.assign(obj)
    if type_key == type_keys.Value.INTEGER:
        binary_data = struct.pack('!q', obj)
    elif type_key == type_keys.Value.FLOAT:
        binary_data = struct.pack('!d', obj)
    elif type_key == type_keys.Value.COMPLEX:
        binary_data = struct.pack(formats.COMPLEX_PACK, obj.real, obj.imag)
    elif type_key == type_keys.Value.NUMPY_OBJ:
        binary_data = common.data_to_binary(obj, np.save)
    elif type_key == type_keys.Value.STRING:
        binary_data = obj.encode(common.ENCODE)
    elif type_key in (type_keys.Value.NULL, type_keys.Value.CASE_DEFAULT):
        binary_data = b''
    elif type_key == type_keys.Value.PARAMETER_VECTOR:
        binary_data = common.data_to_binary(obj, _write_parameter_vec)
    elif type_key == type_keys.Value.PARAMETER:
        binary_data = common.data_to_binary(obj, _write_parameter)
    elif type_key == type_keys.Value.PARAMETER_EXPRESSION:
        binary_data = common.data_to_binary(obj, _write_parameter_expression, use_symengine=use_symengine)
    elif type_key == type_keys.Value.EXPRESSION:
        clbit_indices = {} if index_map is None else index_map['c']
        binary_data = common.data_to_binary(obj, _write_expr, clbit_indices=clbit_indices)
    else:
        raise exceptions.QpyError(f'Serialization for {type_key} is not implemented in value I/O.')
    return (type_key, binary_data)