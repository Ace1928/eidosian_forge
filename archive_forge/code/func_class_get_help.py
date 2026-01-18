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
@classmethod
def class_get_help(cls, inst: HasTraits | None=None) -> str:
    """Get the help string for this class in ReST format.

        If `inst` is given, its current trait values will be used in place of
        class defaults.
        """
    assert inst is None or isinstance(inst, cls)
    final_help = []
    base_classes = ', '.join((p.__name__ for p in cls.__bases__))
    final_help.append(f'{cls.__name__}({base_classes}) options')
    final_help.append(len(final_help[0]) * '-')
    for _, v in sorted(cls.class_traits(config=True).items()):
        help = cls.class_get_trait_help(v, inst)
        final_help.append(help)
    return '\n'.join(final_help)