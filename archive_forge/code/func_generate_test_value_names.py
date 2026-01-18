import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def generate_test_value_names(schema: FunctionSchema, index: int) -> str:
    assert not schema.is_out_fn()
    return ','.join((f'{arg.name}{index}' for arg in schema.schema_order_arguments()))