from xdg.IniFile import IniFile, is_ascii
import xdg.Locale
from xdg.Exceptions import ParsingError
from xdg.util import which
import os.path
import re
import warnings
def findTryExec(self):
    """Looks in the PATH for the executable given in the TryExec field.
        
        Returns the full path to the executable if it is found, None if not.
        Raises :class:`~xdg.Exceptions.NoKeyError` if TryExec is not present.
        """
    tryexec = self.get('TryExec', strict=True)
    return which(tryexec)