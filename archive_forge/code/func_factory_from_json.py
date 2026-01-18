import dataclasses
import datetime
import gzip
import json
import numbers
import pathlib
from typing import (
import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def factory_from_json(type_str: str, resolvers: Optional[Sequence[JsonResolver]]=None) -> ObjectFactory:
    """Returns a factory for constructing objects of type `type_str`.

    DEFAULT_RESOLVERS is updated dynamically as cirq submodules are imported.

    Args:
        type_str: string representation of the type to deserialize.
        resolvers: list of JsonResolvers to use in type resolution. If this is
            left blank, DEFAULT_RESOLVERS will be used.

    Returns:
        An ObjectFactory that can be called to construct an object whose type
        matches the name `type_str`.

    Raises:
        ValueError: if type_str does not have a match in `resolvers`.
    """
    resolvers = resolvers if resolvers is not None else DEFAULT_RESOLVERS
    for resolver in resolvers:
        cirq_type = resolver(type_str)
        if cirq_type is not None:
            return cirq_type
    raise ValueError(f"Could not resolve type '{type_str}' during deserialization")