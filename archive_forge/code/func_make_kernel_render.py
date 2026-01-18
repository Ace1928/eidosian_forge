import functools
import itertools
import logging
from typing import List, Optional
from unittest.mock import patch
import sympy
import torch
from ...autotune_process import CUDABenchmarkRequest, TensorMeta
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout
from ...utils import IndentedBuffer, unique
from ...virtualized import V
from ..common import KernelTemplate
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel
def make_kernel_render(template_node: CUDATemplateBuffer, epilogue_nodes: Optional[List[IRNode]]=None):
    kernel = CUDATemplateKernel(kernel_name='KERNEL_NAME')
    render = functools.partial(self.render, kernel=kernel, template_buffer_node=template_node, epilogue_nodes=epilogue_nodes, **kwargs)
    return (kernel, render)