import sys
from typing import Optional
from traitlets.config import Configurable
from .manager import ActionResult, ExtensionManager, ExtensionPackage  # noqa: F401
from .pypi import PyPIExtensionManager
from .readonly import ReadOnlyExtensionManager
def get_readonly_manager(app_options: Optional[dict]=None, ext_options: Optional[dict]=None, parent: Optional[Configurable]=None) -> ExtensionManager:
    """Read-Only Extension Manager factory"""
    return ReadOnlyExtensionManager(app_options, ext_options, parent)