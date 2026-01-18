from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
def link_extension(self, name):
    """Link an extension by name."""
    linked = self.linked_extensions.get(name, False)
    extension = self.extensions[name]
    if not linked and extension.enabled:
        try:
            extension.link_all_points(self.serverapp)
            self.linked_extensions[name] = True
            self.log.info('%s | extension was successfully linked.', name)
        except Exception as e:
            if self.serverapp and self.serverapp.reraise_server_extension_failures:
                raise
            self.log.warning('%s | error linking extension: %s', name, e, exc_info=True)