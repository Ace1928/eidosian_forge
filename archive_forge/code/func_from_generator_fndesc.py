import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
@classmethod
def from_generator_fndesc(cls, func_ir, fndesc, gentype, mangler):
    """
        Build a GeneratorDescriptor for the generator returned by the
        function described by *fndesc*, with type *gentype*.

        The generator inherits the env_name from the *fndesc*.
        All emitted functions for the generator shares the same Env.
        """
    assert isinstance(gentype, types.Generator)
    restype = gentype.yield_type
    args = ['gen']
    argtypes = (gentype,)
    qualname = fndesc.qualname + '.next'
    unique_name = fndesc.unique_name + '.next'
    self = cls(fndesc.native, fndesc.modname, qualname, unique_name, fndesc.doc, fndesc.typemap, restype, fndesc.calltypes, args, fndesc.kws, argtypes=argtypes, mangler=mangler, inline=False, env_name=fndesc.env_name)
    return self