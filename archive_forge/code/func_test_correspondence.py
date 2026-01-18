import numpy as np
import pytest
import sympy
from google.protobuf import json_format
import cirq_google
from cirq_google.serialization.arg_func_langs import (
from cirq_google.api import v2
@pytest.mark.parametrize('min_lang,value,proto', [('', 1.0, {'arg_value': {'float_value': 1.0}}), ('', 1, {'arg_value': {'float_value': 1.0}}), ('', 'abc', {'arg_value': {'string_value': 'abc'}}), ('', [True, False], {'arg_value': {'bool_values': {'values': [True, False]}}}), ('', [42.9, 3.14], {'arg_value': {'double_values': {'values': [42.9, 3.14]}}}), ('', [3, 8], {'arg_value': {'int64_values': {'values': ['3', '8']}}}), ('', ['t1', 't2'], {'arg_value': {'string_values': {'values': ['t1', 't2']}}}), ('', sympy.Symbol('x'), {'symbol': 'x'}), ('linear', sympy.Symbol('x') - sympy.Symbol('y'), {'func': {'type': 'add', 'args': [{'symbol': 'x'}, {'func': {'type': 'mul', 'args': [{'arg_value': {'float_value': -1.0}}, {'symbol': 'y'}]}}]}}), ('exp', sympy.Symbol('x') ** sympy.Symbol('y'), {'func': {'type': 'pow', 'args': [{'symbol': 'x'}, {'symbol': 'y'}]}})])
def test_correspondence(min_lang: str, value: ARG_LIKE, proto: v2.program_pb2.Arg):
    msg = v2.program_pb2.Arg()
    json_format.ParseDict(proto, msg)
    min_i = LANGUAGE_ORDER.index(min_lang)
    for i, lang in enumerate(LANGUAGE_ORDER):
        if i < min_i:
            with pytest.raises(ValueError, match='not supported by arg_function_language'):
                _ = arg_to_proto(value, arg_function_language=lang)
            with pytest.raises(ValueError, match='Unrecognized function type'):
                _ = arg_from_proto(msg, arg_function_language=lang)
        else:
            parsed = arg_from_proto(msg, arg_function_language=lang)
            packed = json_format.MessageToDict(arg_to_proto(value, arg_function_language=lang), including_default_value_fields=True, preserving_proto_field_name=True, use_integers_for_enums=True)
            assert parsed == value
            assert packed == proto