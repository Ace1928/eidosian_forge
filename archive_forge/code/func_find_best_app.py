from __future__ import annotations
import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType
import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv
def find_best_app(module: ModuleType) -> Flask:
    """Given a module instance this tries to find the best possible
    application in the module or raises an exception.
    """
    from . import Flask
    for attr_name in ('app', 'application'):
        app = getattr(module, attr_name, None)
        if isinstance(app, Flask):
            return app
    matches = [v for v in module.__dict__.values() if isinstance(v, Flask)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise NoAppException(f"Detected multiple Flask applications in module '{module.__name__}'. Use '{module.__name__}:name' to specify the correct one.")
    for attr_name in ('create_app', 'make_app'):
        app_factory = getattr(module, attr_name, None)
        if inspect.isfunction(app_factory):
            try:
                app = app_factory()
                if isinstance(app, Flask):
                    return app
            except TypeError as e:
                if not _called_with_wrong_args(app_factory):
                    raise
                raise NoAppException(f"Detected factory '{attr_name}' in module '{module.__name__}', but could not call it without arguments. Use '{module.__name__}:{attr_name}(args)' to specify arguments.") from e
    raise NoAppException(f"Failed to find Flask application or factory in module '{module.__name__}'. Use '{module.__name__}:name' to specify one.")