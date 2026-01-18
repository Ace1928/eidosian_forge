import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath
def replaceSysModules(self, sysModules: Iterable[Tuple[str, ModuleType]]) -> None:
    """
        Replace sys.modules, for the duration of the test, with the given value.
        """
    originalSysModules = sys.modules.copy()

    def cleanUpSysModules() -> None:
        sys.modules.clear()
        sys.modules.update(originalSysModules)
    self.addCleanup(cleanUpSysModules)
    sys.modules.clear()
    sys.modules.update(sysModules)