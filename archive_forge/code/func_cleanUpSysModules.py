import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath
def cleanUpSysModules() -> None:
    sys.modules.clear()
    sys.modules.update(originalSysModules)