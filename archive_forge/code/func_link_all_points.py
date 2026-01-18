from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
def link_all_points(self, serverapp):
    """Link all extension points."""
    for point_name in self.extension_points:
        self.link_point(point_name, serverapp)