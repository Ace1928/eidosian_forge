from __future__ import annotations
import json
from pathlib import Path, PurePath
from typing import Any, Dict, Union
from jsonschema import FormatChecker, validators
from referencing import Registry
from referencing.jsonschema import DRAFT7
from . import yaml
from .validators import draft7_format_checker, validate_schema
def intended_as_path(schema: str) -> bool:
    path = Path(schema)
    return path.match('*.yml') or path.match('*.yaml') or path.match('*.json')