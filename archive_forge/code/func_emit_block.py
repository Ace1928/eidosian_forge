import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def emit_block(decls):
    return '\n.. rst-class:: codeblock-height-limiter\n\n::\n\n{}\n'.format(''.join((f'    {d}\n\n' for d in decls)))