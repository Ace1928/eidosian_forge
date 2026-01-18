import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode
from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
def construct_operators(operator_list_from_yaml: List[Any]) -> str:
    operator_list_part = []
    for operator in operator_list_from_yaml:
        operator_list_part.append(ONE_OPERATOTR_STRING.substitute(operator_name=operator[0], overload_name=operator[1], num_of_args=operator[2]))
    return OPERATOR_STRING_LIST.substitute(operator_string_list=''.join(operator_list_part).lstrip('\n'))