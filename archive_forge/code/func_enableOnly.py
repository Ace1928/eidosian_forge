from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar
import warnings
from markdown_it._compat import DATACLASS_KWARGS
from .utils import EnvType
def enableOnly(self, names: str | Iterable[str], ignoreInvalid: bool=False) -> list[str]:
    """Enable rules with given names, and disable everything else.

        :param names: name or list of rule names to enable.
        :param ignoreInvalid: ignore errors when rule not found
        :raises: KeyError if name not found and not ignoreInvalid
        :return: list of found rule names
        """
    if isinstance(names, str):
        names = [names]
    for rule in self.__rules__:
        rule.enabled = False
    return self.enable(names, ignoreInvalid)