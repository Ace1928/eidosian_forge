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
def json_namespace(type_obj: Type) -> str:
    """Returns a namespace for JSON serialization of `type_obj`.

    Types can provide custom namespaces with `_json_namespace_`; otherwise, a
    Cirq type will not include a namespace in its cirq_type. Non-Cirq types
    must provide a namespace for serialization in Cirq.

    Args:
        type_obj: Type to retrieve the namespace from.

    Returns:
        The namespace to prepend `type_obj` with in its JSON cirq_type.

    Raises:
        ValueError: if `type_obj` is not a Cirq type and does not explicitly
            define its namespace with _json_namespace_.
    """
    if hasattr(type_obj, '_json_namespace_'):
        return type_obj._json_namespace_()
    if type_obj.__module__.startswith('cirq'):
        return ''
    raise ValueError(f'{type_obj} is not a Cirq type, and does not define _json_namespace_.')