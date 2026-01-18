import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer
from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .cutlass_epilogue_gen import (
def gemm_mode(self) -> str:
    sizes = self.output_node.get_size()
    if len(sizes) > 2:
        return 'cutlass::gemm::GemmUniversalMode::kBatched'
    else:
        return 'cutlass::gemm::GemmUniversalMode::kGemm'