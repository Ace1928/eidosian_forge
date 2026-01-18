import inspect
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Type, TypeVar, Union, get_args
def normalize_key(key: str) -> str:
    return key.replace('-', '_').replace(' ', '_').lower()