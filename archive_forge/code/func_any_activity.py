from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
def any_activity(self):
    """Check for any activity currently happening across all extension applications."""
    for _, apps in sorted(dict(self.extension_apps).items()):
        for app in apps:
            if app.current_activity():
                return True