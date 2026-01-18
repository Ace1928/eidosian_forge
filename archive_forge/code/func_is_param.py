import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def is_param(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a parameter within the exported program
    """
    return node.name in program.graph_signature.inputs_to_parameters