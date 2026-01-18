import builtins as builtin_mod
from traitlets.config.configurable import Configurable
from traitlets import Instance
def remove_builtin(self, key, orig):
    """Remove an added builtin and re-set the original."""
    if orig is BuiltinUndefined:
        del builtin_mod.__dict__[key]
    else:
        builtin_mod.__dict__[key] = orig