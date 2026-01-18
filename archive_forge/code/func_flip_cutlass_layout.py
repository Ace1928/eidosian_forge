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
@staticmethod
def flip_cutlass_layout(cutlass_layout: 'cutlass_lib.LayoutType') -> 'cutlass_lib.LayoutType':
    assert cutlass_utils.try_import_cutlass()
    import cutlass_library.library as cutlass_lib
    if cutlass_layout == cutlass_lib.LayoutType.RowMajor:
        return cutlass_lib.LayoutType.ColumnMajor
    else:
        return cutlass_lib.LayoutType.RowMajor