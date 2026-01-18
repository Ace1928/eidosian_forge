from __future__ import annotations
from contextlib import suppress
from datetime import date, datetime
from uuid import UUID
import ipaddress
import re
import typing
import warnings
from jsonschema.exceptions import FormatError
@_checks_drafts(draft6='uri-template', draft7='uri-template', draft201909='uri-template', draft202012='uri-template')
def is_uri_template(instance: object) -> bool:
    if not isinstance(instance, str):
        return True
    return uri_template.validate(instance)