from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
@property
def extension_points(self):
    """Return mapping of extension point names and ExtensionPoint objects."""
    return {name: point for value in self.extensions.values() for name, point in value.extension_points.items()}