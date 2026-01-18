import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_test_value_definitions(schema: FunctionSchema, index: int) -> str:
    assert not schema.is_out_fn()
    schema_name = schema.name.name.base
    arg_map = {}
    for arg in schema.schema_order_arguments():
        test_value_exp = test_value_expression(arg.type, index, schema_name)
        arg_map[arg.name] = test_value_exp
    config.override_test_values(arg_map, schema_name, index)
    arg_populations = []
    for arg_name, arg_value in arg_map.items():
        arg_populations.append(f'auto {arg_name}{index} = {arg_value}')
    return ';\n    '.join(arg_populations) + ';'