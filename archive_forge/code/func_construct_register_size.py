import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.jit.generate_bytecode import generate_upgraders_bytecode
from torchgen.code_template import CodeTemplate
from torchgen.operator_versions.gen_mobile_upgraders_constant import (
def construct_register_size(register_size_from_yaml: int) -> str:
    if not isinstance(register_size_from_yaml, int):
        raise ValueError(f"Input register size is {register_size_from_yaml} andit's type is {{type(register_size_from_yaml)}}. An int type is expected.")
    return str(register_size_from_yaml)