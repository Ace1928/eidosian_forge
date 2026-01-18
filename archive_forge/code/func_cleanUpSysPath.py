import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath
def cleanUpSysPath() -> None:
    sys.path[:] = originalSysPath