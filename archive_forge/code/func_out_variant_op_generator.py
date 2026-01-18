import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def out_variant_op_generator(self, g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
    functional = g.functional
    schema = str(functional.func)
    populated_argument = generate_arg_extraction(g.functional.func)
    functional_variant_call = generate_non_out_variant_call(g, backend_index)
    assert len(g.out.func.arguments.out) == 1
    out_variable_name = str(g.out.func.arguments.out[0].name)
    out_variant_call = generate_out_variant_call(g, backend_index)
    generated = f'\n      if (n->matches(torch::schema("aten::{schema}"))) {{\n        return [](ProcessedNode* p_node) {{\n          {populated_argument}\n          if (p_node->Output(0).isNone()) {{\n            p_node->Output(0) = {functional_variant_call};\n            return;\n          }}\n          auto& {out_variable_name} = p_node->Output(0).toTensor();\n          fastResizeToZero({out_variable_name});\n          {out_variant_call};\n        }};\n      }}'
    return generated