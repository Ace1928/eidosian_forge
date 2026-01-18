import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode
from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
def get_upgrader_bytecode_function_to_index_map(upgrader_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
    upgrader_bytecode_function_to_index_map = {}
    index = 0
    for upgrader_bytecode in upgrader_dict:
        for upgrader_name in upgrader_bytecode.keys():
            if upgrader_name in EXCLUE_UPGRADER_SET:
                continue
            upgrader_bytecode_function_to_index_map[upgrader_name] = index
            index += 1
    return upgrader_bytecode_function_to_index_map