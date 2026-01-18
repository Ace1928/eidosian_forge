from __future__ import annotations
import configparser
import importlib.metadata
import inspect
import itertools
import logging
import sys
from typing import Any
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from flake8 import utils
from flake8.defaults import VALID_CODE_PREFIX
from flake8.exceptions import ExecutionError
from flake8.exceptions import FailedToLoadPlugin
def parse_plugin_options(cfg: configparser.RawConfigParser, cfg_dir: str, *, enable_extensions: str | None, require_plugins: str | None) -> PluginOptions:
    """Parse plugin loading related options."""
    paths_s = cfg.get('flake8:local-plugins', 'paths', fallback='').strip()
    paths = utils.parse_comma_separated_list(paths_s)
    paths = utils.normalize_paths(paths, cfg_dir)
    return PluginOptions(local_plugin_paths=tuple(paths), enable_extensions=frozenset(_parse_option(cfg, 'enable_extensions', enable_extensions)), require_plugins=frozenset(_parse_option(cfg, 'require_plugins', require_plugins)))