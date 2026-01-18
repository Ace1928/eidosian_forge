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
def remove_modifier(self, *, schema_id: str | None=None, modifier: t.Callable[[str, dict[str, t.Any]], dict[str, t.Any]]) -> None:
    """Remove a modifier from an event or all events.

        Parameters
        ----------
        schema_id: str
            If given, remove this modifier only for a specific event type.
        modifier: Callable[[str, dict], dict]

            The modifier to remove.
        """
    if schema_id:
        self._modifiers[schema_id].discard(modifier)
    else:
        for schema_id in self.schemas.schema_ids:
            self._modifiers[schema_id].discard(modifier)
            self._modifiers[schema_id].discard(modifier)