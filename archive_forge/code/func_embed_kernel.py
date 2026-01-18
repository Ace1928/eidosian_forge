import sys
from .core.getipython import get_ipython
from .core import release
from .core.application import Application
from .terminal.embed import embed
from .core.interactiveshell import InteractiveShell
from .utils.sysinfo import sys_info
from .utils.frame import extract_module_locals
def embed_kernel(module=None, local_ns=None, **kwargs):
    """Embed and start an IPython kernel in a given scope.

    If you don't want the kernel to initialize the namespace
    from the scope of the surrounding function,
    and/or you want to load full IPython configuration,
    you probably want `IPython.start_kernel()` instead.

    Parameters
    ----------
    module : types.ModuleType, optional
        The module to load into IPython globals (default: caller)
    local_ns : dict, optional
        The namespace to load into IPython user namespace (default: caller)
    **kwargs : various, optional
        Further keyword args are relayed to the IPKernelApp constructor,
        such as `config`, a traitlets :class:`Config` object (see :ref:`configure_start_ipython`),
        allowing configuration of the kernel (see :ref:`kernel_options`).  Will only have an effect
        on the first embed_kernel call for a given process.
    """
    caller_module, caller_locals = extract_module_locals(1)
    if module is None:
        module = caller_module
    if local_ns is None:
        local_ns = caller_locals
    from ipykernel.embed import embed_kernel as real_embed_kernel
    real_embed_kernel(module=module, local_ns=local_ns, **kwargs)