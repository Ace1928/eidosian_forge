from __future__ import annotations
import logging # isort:skip
import platform
import sys
from argparse import Namespace
from bokeh import __version__
from bokeh.settings import settings
from bokeh.util.compiler import nodejs_version, npmjs_version
from bokeh.util.dependencies import import_optional
from ..subcommand import Argument, Subcommand
def if_installed(version_or_none: str | None) -> str:
    """ helper method to optionally return module version number or not installed

    :param version_or_none:
    :return:
    """
    return version_or_none or '(not installed)'