import sys
from typing import Optional
from traitlets.config import Configurable
from .manager import ActionResult, ExtensionManager, ExtensionPackage  # noqa: F401
from .pypi import PyPIExtensionManager
from .readonly import ReadOnlyExtensionManager
def get_pypi_manager(app_options: Optional[dict]=None, ext_options: Optional[dict]=None, parent: Optional[Configurable]=None) -> ExtensionManager:
    """PyPi Extension Manager factory"""
    return PyPIExtensionManager(app_options, ext_options, parent)