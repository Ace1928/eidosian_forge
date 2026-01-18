from __future__ import annotations
import asyncio
import csv
import io
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from jupyter_server.base.handlers import APIHandler
from tornado import web
from traitlets import List, Unicode
from traitlets.config import LoggingConfigurable
from .config import get_federated_extensions
@property
def federated_extensions(self) -> dict[str, Any]:
    """Lazily load the currrently-available federated extensions.

        This is expensive, but probably the only way to be sure to get
        up-to-date license information for extensions installed interactively.
        """
    if TYPE_CHECKING:
        from .app import LabServerApp
        assert isinstance(self.parent, LabServerApp)
    per_paths = [self.parent.labextensions_path, self.parent.extra_labextensions_path]
    labextensions_path = [extension for extensions in per_paths for extension in extensions]
    return get_federated_extensions(labextensions_path)