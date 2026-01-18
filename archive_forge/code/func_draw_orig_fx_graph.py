import collections
import contextlib
import cProfile
import dataclasses
import functools
import itertools
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
import torch
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro, wrap_compiler_debug
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._pytree import tree_map
from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
from .virtualized import V
from torch._inductor.debug import load_args_and_run_compile_fx_inner
def draw_orig_fx_graph(self, gm: torch.fx.GraphModule, nodes: SchedulerNodeList):
    annotate_orig_fx_with_snodes(gm, nodes)
    draw_graph(gm, fname=self.filename('orig_fx_graph_diagram.svg'), clear_meta=False, prog=GRAPHVIZ_COMMAND_SCALABLE, parse_stack_trace=True, dot_graph_shape=config.trace.dot_graph_shape)