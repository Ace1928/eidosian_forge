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
@setupmethod
def register_error_handler(self, code_or_exception: type[Exception] | int, f: ft.ErrorHandlerCallable) -> None:
    """Alternative error attach function to the :meth:`errorhandler`
        decorator that is more straightforward to use for non decorator
        usage.

        .. versionadded:: 0.7
        """
    exc_class, code = self._get_exc_class_and_code(code_or_exception)
    self.error_handler_spec[None][code][exc_class] = f