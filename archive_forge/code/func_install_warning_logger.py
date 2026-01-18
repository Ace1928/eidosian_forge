import logging
import warnings
from typing import Any, Optional, TextIO, Type, Union
from pip._vendor.packaging.version import parse
from pip import __version__ as current_version  # NOTE: tests patch this name.
def install_warning_logger() -> None:
    warnings.simplefilter('default', PipDeprecationWarning, append=True)
    global _original_showwarning
    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning
        warnings.showwarning = _showwarning