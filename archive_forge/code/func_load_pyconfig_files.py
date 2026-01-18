from __future__ import annotations
import argparse
import copy
import functools
import json
import os
import re
import sys
import typing as t
from logging import Logger
from traitlets.traitlets import Any, Container, Dict, HasTraits, List, TraitType, Undefined
from ..utils import cast_unicode, filefind, warnings
def load_pyconfig_files(config_files: list[str], path: str) -> Config:
    """Load multiple Python config files, merging each of them in turn.

    Parameters
    ----------
    config_files : list of str
        List of config files names to load and merge into the config.
    path : unicode
        The full path to the location of the config files.
    """
    config = Config()
    for cf in config_files:
        loader = PyFileConfigLoader(cf, path=path)
        try:
            next_config = loader.load_config()
        except ConfigFileNotFound:
            pass
        except Exception:
            raise
        else:
            config.merge(next_config)
    return config