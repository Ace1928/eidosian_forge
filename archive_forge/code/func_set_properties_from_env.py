from __future__ import annotations
import os
import abc
import logging
import operator
import copy
import typing
from .py312compat import metadata
from . import credentials, errors, util
from ._compat import properties
def set_properties_from_env(self) -> None:
    """For all KEYRING_PROPERTY_* env var, set that property."""

    def parse(item: typing.Tuple[str, str]):
        key, value = item
        pre, sep, name = key.partition('KEYRING_PROPERTY_')
        return sep and (name.lower(), value)
    props: filter[typing.Tuple[str, str]] = filter(None, map(parse, os.environ.items()))
    for name, value in props:
        setattr(self, name, value)