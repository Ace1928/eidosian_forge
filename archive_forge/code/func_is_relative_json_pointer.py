from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@_checks_drafts(draft7='relative-json-pointer', draft201909='relative-json-pointer', draft202012='relative-json-pointer', raises=jsonpointer.JsonPointerException)
def is_relative_json_pointer(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    if not instance:
        return False
    non_negative_integer, rest = ([], '')
    for i, character in enumerate(instance):
        if character.isdigit():
            if i > 0 and int(instance[i - 1]) == 0:
                return False
            non_negative_integer.append(character)
            continue
        if not non_negative_integer:
            return False
        rest = instance[i:]
        break
    return rest == '#' or bool(jsonpointer.JsonPointer(rest))