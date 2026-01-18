from llvmlite import ir
from numba.core.typing.templates import ConcreteTemplate
from numba.core import types, typing, funcdesc, config, compiler, sigutils
from numba.core.compiler import (sanitize_compile_result_entries, CompilerBase,
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass,
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.typed_passes import (IRLegalization, NativeLowering,
from warnings import warn
from numba.cuda.api import get_current_device
from numba.cuda.target import CUDACABICallConv
def compile_ptx_for_current_device(pyfunc, sig, debug=False, lineinfo=False, device=False, fastmath=False, opt=True, abi='numba', abi_info=None):
    """Compile a Python function to PTX for a given set of argument types for
    the current device's compute capabilility. This calls :func:`compile_ptx`
    with an appropriate ``cc`` value for the current device."""
    cc = get_current_device().compute_capability
    return compile_ptx(pyfunc, sig, debug=debug, lineinfo=lineinfo, device=device, fastmath=fastmath, cc=cc, opt=opt, abi=abi, abi_info=abi_info)