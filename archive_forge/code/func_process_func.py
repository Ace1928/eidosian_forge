import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def process_func(func: Callable):
    if func not in script_func_map:
        scripted_func = torch.jit.script(func)
        torch._C._jit_pass_inline(scripted_func.graph)
        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)
        script_func_map[func] = scripted_func
    return script_func_map[func]