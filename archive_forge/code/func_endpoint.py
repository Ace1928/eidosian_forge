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
def endpoint(self, endpoint: str) -> t.Callable[[F], F]:
    """Decorate a view function to register it for the given
        endpoint. Used if a rule is added without a ``view_func`` with
        :meth:`add_url_rule`.

        .. code-block:: python

            app.add_url_rule("/ex", endpoint="example")

            @app.endpoint("example")
            def example():
                ...

        :param endpoint: The endpoint name to associate with the view
            function.
        """

    def decorator(f: F) -> F:
        self.view_functions[endpoint] = f
        return f
    return decorator