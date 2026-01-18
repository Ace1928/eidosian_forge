from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
def notice_config_override(change: Bunch) -> None:
    """Record traits set by both config and kwargs.

            They will need to be overridden again after loading config.
            """
    if change.name in kwargs:
        config_override_names.add(change.name)