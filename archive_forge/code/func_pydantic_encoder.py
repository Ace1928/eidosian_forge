import datetime
from collections import deque
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from re import Pattern
from types import GeneratorType
from typing import Any, Callable, Dict, Type, Union
from uuid import UUID
from .color import Color
from .networks import NameEmail
from .types import SecretBytes, SecretStr
def pydantic_encoder(obj: Any) -> Any:
    from dataclasses import asdict, is_dataclass
    from .main import BaseModel
    if isinstance(obj, BaseModel):
        return obj.dict()
    elif is_dataclass(obj):
        return asdict(obj)
    for base in obj.__class__.__mro__[:-1]:
        try:
            encoder = ENCODERS_BY_TYPE[base]
        except KeyError:
            continue
        return encoder(obj)
    else:
        raise TypeError(f"Object of type '{obj.__class__.__name__}' is not JSON serializable")