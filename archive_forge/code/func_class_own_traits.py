from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
@classmethod
def class_own_traits(cls: type[HasTraits], **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
    """Get a dict of all the traitlets defined on this class, not a parent.

        Works like `class_traits`, except for excluding traits from parents.
        """
    sup = super(cls, cls)
    return {n: t for n, t in cls.class_traits(**metadata).items() if getattr(sup, n, None) is not t}