import uuid
import json
import typing
import codecs
import hashlib
import datetime
import contextlib
import dataclasses
from enum import Enum
from .lazy import lazy_import, get_obj_class_name
def object_deserializer(obj: typing.Dict) -> typing.Dict:
    results = {}
    for key, value in obj.items():
        if isinstance(value, bytes):
            encoding = guess_json_utf(value)
            results[key] = value.decode(encoding) if encoding is not None else value
            continue
        if isinstance(value, dict):
            results[key] = object_deserializer(value)
            continue
        for dt_key in {'created', 'updated', 'modified', 'timestamp', 'date'}:
            if dt_key in key:
                if isinstance(value, str):
                    with contextlib.suppress(Exception):
                        results[key] = datetime.datetime.strptime(value, '%a, %d %b %Y %H:%M:%S GMT')
                        continue
                    with contextlib.suppress(Exception):
                        results[key] = datetime.datetime.fromisoformat(value)
                        continue
                with contextlib.suppress(Exception):
                    results[key] = datetime.datetime.fromtimestamp(value, tz=datetime.timezone.utc)
                    continue
        results[key] = value
    return results