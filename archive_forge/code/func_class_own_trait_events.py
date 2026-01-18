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
def class_own_trait_events(cls: type[HasTraits], name: str) -> dict[str, EventHandler]:
    """Get a dict of all event handlers defined on this class, not a parent.

        Works like ``event_handlers``, except for excluding traits from parents.
        """
    sup = super(cls, cls)
    return {n: e for n, e in cls.events(name).items() if getattr(sup, n, None) is not e}