import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_arg_extraction(schema: FunctionSchema) -> str:
    arg_populations = []
    for i, arg in enumerate(schema.schema_order_arguments()):
        maybe_method = ivalue_type_conversion_method(arg.type)
        assert maybe_method
        is_reference, type_conversion_method = maybe_method
        reference = '&' if is_reference else ''
        arg_populations.append(f'const auto{reference} {arg.name} = p_node->Input({i}).{type_conversion_method}')
    return ';\n    '.join(arg_populations) + ';'