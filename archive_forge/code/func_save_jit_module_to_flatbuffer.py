import os
import pathlib
import torch
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device
def save_jit_module_to_flatbuffer(m, f, _extra_files=None):
    """
    Save an offline version of this module for use in a separate process.

    The saved module serializes all of the methods, submodules, parameters, and
    attributes of this module. It can be loaded into the C++ API using
    ``torch::jit::load_jit_module_from_file(filename)`` or into the Python API with
    :func:`torch.jit.jit_module_from_flatbuffer<torch.jit.jit_module_from_flatbuffer>`.

    To be able to save a module, it must not make any calls to native Python
    functions.  This means that all submodules must be subclasses of
    :class:`ScriptModule` as well.

    .. DANGER::
        All modules, no matter their device, are always loaded onto the CPU
        during loading.  This is different from :func:`torch.load`'s semantics
        and may change in the future.

    Args:
        m: A :class:`ScriptModule` to save.
        f: A string for file path


    Example:
    .. testcode::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        m = torch.jit.script(MyModule())

        # Save to file
        torch.jit.save_jit_module_to_flatbuffer(m, 'scriptmodule.ff')
    """
    extra_files = _extra_files
    if extra_files is None:
        extra_files = {}
    if isinstance(f, (str, pathlib.Path)):
        f = str(f)
        torch._C._save_jit_module(m._c, f, extra_files)
    else:
        s = torch._C._save_jit_module_to_bytes(m._c, extra_files)
        f.write(s)