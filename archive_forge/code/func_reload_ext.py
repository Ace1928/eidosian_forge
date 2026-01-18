from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
@line_magic
def reload_ext(self, module_str):
    """Reload an IPython extension by its module name."""
    if not module_str:
        raise UsageError('Missing module name.')
    self.shell.extension_manager.reload_extension(module_str)