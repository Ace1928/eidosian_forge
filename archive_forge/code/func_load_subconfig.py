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
def load_subconfig(self, fname: str, path: str | None=None) -> None:
    """Injected into config file namespace as load_subconfig"""
    if path is None:
        path = self.path
    loader = self.__class__(fname, path)
    try:
        sub_config = loader.load_config()
    except ConfigFileNotFound:
        pass
    else:
        self.config.merge(sub_config)