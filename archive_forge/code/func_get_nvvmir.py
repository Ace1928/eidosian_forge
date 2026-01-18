import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
def get_nvvmir(self):
    versions = NVVM().get_ir_version()
    data_layout = NVVM().data_layout
    return nvvmir_generic.format(data_layout=data_layout, v=versions)