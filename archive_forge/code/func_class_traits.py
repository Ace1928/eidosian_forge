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
def class_traits(cls: type[HasTraits], **metadata: t.Any) -> dict[str, TraitType[t.Any, t.Any]]:
    """Get a ``dict`` of all the traits of this class.  The dictionary
        is keyed on the name and the values are the TraitType objects.

        This method is just like the :meth:`traits` method, but is unbound.

        The TraitTypes returned don't know anything about the values
        that the various HasTrait's instances are holding.

        The metadata kwargs allow functions to be passed in which
        filter traits based on metadata values.  The functions should
        take a single value as an argument and return a boolean.  If
        any function returns False, then the trait is not included in
        the output.  If a metadata key doesn't exist, None will be passed
        to the function.
        """
    traits = cls._traits.copy()
    if len(metadata) == 0:
        return traits
    result = {}
    for name, trait in traits.items():
        for meta_name, meta_eval in metadata.items():
            if not callable(meta_eval):
                meta_eval = _SimpleTest(meta_eval)
            if not meta_eval(trait.metadata.get(meta_name, None)):
                break
        else:
            result[name] = trait
    return result