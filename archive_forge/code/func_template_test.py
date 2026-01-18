from __future__ import annotations
import logging
import os
import sys
import typing as t
from datetime import timedelta
from itertools import chain
from werkzeug.exceptions import Aborter
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.routing import BuildError
from werkzeug.routing import Map
from werkzeug.routing import Rule
from werkzeug.sansio.response import Response
from werkzeug.utils import cached_property
from werkzeug.utils import redirect as _wz_redirect
from .. import typing as ft
from ..config import Config
from ..config import ConfigAttribute
from ..ctx import _AppCtxGlobals
from ..helpers import _split_blueprint_path
from ..helpers import get_debug_flag
from ..json.provider import DefaultJSONProvider
from ..json.provider import JSONProvider
from ..logging import create_logger
from ..templating import DispatchingJinjaLoader
from ..templating import Environment
from .scaffold import _endpoint_from_view_func
from .scaffold import find_package
from .scaffold import Scaffold
from .scaffold import setupmethod
@setupmethod
def template_test(self, name: str | None=None) -> t.Callable[[T_template_test], T_template_test]:
    """A decorator that is used to register custom template test.
        You can specify a name for the test, otherwise the function
        name will be used. Example::

          @app.template_test()
          def is_prime(n):
              if n == 2:
                  return True
              for i in range(2, int(math.ceil(math.sqrt(n))) + 1):
                  if n % i == 0:
                      return False
              return True

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

    def decorator(f: T_template_test) -> T_template_test:
        self.add_template_test(f, name=name)
        return f
    return decorator