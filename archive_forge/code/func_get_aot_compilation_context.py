import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def get_aot_compilation_context() -> Tuple[List[str], str, int]:
    return (list(graph_being_compiled), model_name, nth_graph)