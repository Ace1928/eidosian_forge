from __future__ import annotations
import asyncio
import copy
import json
import logging
import typing as t
import warnings
from datetime import datetime, timezone
from jsonschema import ValidationError
from pythonjsonlogger import jsonlogger
from traitlets import Dict, Instance, Set, default
from traitlets.config import Config, LoggingConfigurable
from .schema import SchemaType
from .schema_registry import SchemaRegistry
from .traits import Handlers
from .validators import JUPYTER_EVENTS_CORE_VALIDATOR
def add_modifier(self, *, schema_id: str | None=None, modifier: t.Callable[[str, dict[str, t.Any]], dict[str, t.Any]]) -> None:
    """Add a modifier (callable) to a registered event.

        Parameters
        ----------
        modifier: Callable
            A callable function/method that executes when the named event occurs.
            This method enforces a string signature for modifiers:

                (schema_id: str, data: dict) -> dict:
        """
    if not callable(modifier):
        msg = '`modifier` must be a callable'
        raise TypeError(msg)
    if schema_id:
        modifiers = self._modifiers.get(schema_id, set())
        modifiers.add(modifier)
        self._modifiers[schema_id] = modifiers
        return
    for id_ in self._modifiers:
        if schema_id is None or id_ == schema_id:
            self._modifiers[id_].add(modifier)