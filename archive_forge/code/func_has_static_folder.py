from __future__ import annotations
import importlib.util
import os
import pathlib
import sys
import typing as t
from collections import defaultdict
from functools import update_wrapper
from jinja2 import BaseLoader
from jinja2 import FileSystemLoader
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug.utils import cached_property
from .. import typing as ft
from ..helpers import get_root_path
from ..templating import _default_template_ctx_processor
@property
def has_static_folder(self) -> bool:
    """``True`` if :attr:`static_folder` is set.

        .. versionadded:: 0.5
        """
    return self.static_folder is not None