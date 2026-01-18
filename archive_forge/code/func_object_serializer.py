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
def object_serializer(obj: typing.Any) -> typing.Any:
    if isinstance(obj, dict):
        return {k: object_serializer(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [object_serializer(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode('utf-8')
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if hasattr(obj, 'dict'):
        return obj.dict()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, 'todict'):
        return obj.todict()
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    if hasattr(obj, 'tojson'):
        return obj.tojson()
    if hasattr(obj, 'toJson'):
        return obj.toJson()
    if hasattr(obj, 'json'):
        return obj.json()
    if hasattr(obj, 'encode'):
        return obj.encode()
    if hasattr(obj, 'get_secret_value'):
        return obj.get_secret_value()
    if hasattr(obj, 'as_posix'):
        return obj.as_posix()
    if hasattr(obj, 'numpy'):
        return obj.numpy().tolist()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (datetime.date, datetime.datetime)) or hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if np is not None:
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
    if isinstance(obj, object):
        with contextlib.suppress(Exception):
            return {k: object_serializer(v) for k, v in obj.__dict__.items()}
    with contextlib.suppress(Exception):
        return int(obj)
    with contextlib.suppress(Exception):
        return float(obj)
    with contextlib.suppress(Exception):
        return str(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')