from __future__ import annotations
import logging # isort:skip
import argparse
import os
from fnmatch import fnmatch
from glob import glob
from typing import TYPE_CHECKING, Any
from tornado.autoreload import watch
from bokeh.application import Application
from bokeh.resources import DEFAULT_SERVER_PORT
from bokeh.server.auth_provider import AuthModule, NullAuth
from bokeh.server.tornado import (
from bokeh.settings import settings
from bokeh.util.logconfig import basicConfig
from bokeh.util.strings import format_docstring, nice_join
from ..subcommand import Argument, Subcommand
from ..util import build_single_handler_applications, die, report_server_init_errors
def add_optional_autoreload_files(file_list: list[str]) -> None:
    for filen in file_list:
        if os.path.isdir(filen):
            log.warning('Cannot watch directory ' + filen)
            continue
        log.info('Watching: ' + filen)
        watch(filen)