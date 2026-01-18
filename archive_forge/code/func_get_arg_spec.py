import operator
import torch
from torch.export.exported_program import ConstantArgument, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult
def get_arg_spec(arg):
    if isinstance(arg, torch.fx.Node):
        if isinstance(arg.meta.get('val'), torch.Tensor):
            return TensorArgument(name=arg.name)
        else:
            raise AssertionError('Symint input is not implemented yet for submodule call signature.')
    else:
        return ConstantArgument(value=arg)