from xdg.IniFile import IniFile, is_ascii
import xdg.Locale
from xdg.Exceptions import ParsingError
from xdg.util import which
import os.path
import re
import warnings
def checkCategorie(self, value):
    """Deprecated alias for checkCategories - only exists for backwards
        compatibility.
        """
    warnings.warn('checkCategorie is deprecated, use checkCategories', DeprecationWarning)
    return self.checkCategories(value)