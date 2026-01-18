from __future__ import annotations
import sys
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import TypeVar
import_prefix = _reg.import_prefix
def import_prefix(self, path: str) -> None:
    """Resolve all the modules in the registry that start with the
        specified path.
        """
    for module in self.module_registry:
        if self.prefix:
            key = module.split(self.prefix)[-1].replace('.', '_')
        else:
            key = module
        if (not path or module.startswith(path)) and key not in self.__dict__:
            __import__(module, globals(), locals())
            self.__dict__[key] = globals()[key] = sys.modules[module]