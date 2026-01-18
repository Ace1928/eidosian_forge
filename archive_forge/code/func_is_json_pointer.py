from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@_checks_drafts(draft6='json-pointer', draft7='json-pointer', draft201909='json-pointer', draft202012='json-pointer', raises=jsonpointer.JsonPointerException)
def is_json_pointer(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    return bool(jsonpointer.JsonPointer(instance))