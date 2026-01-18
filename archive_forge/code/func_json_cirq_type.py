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
def json_cirq_type(type_obj: Type) -> str:
    """Returns a string type for JSON serialization of `type_obj`.

    This method is not part of the base serialization path. Together with
    `cirq_type_from_json`, it can be used to provide type-object serialization
    for classes that need it.
    """
    namespace = json_namespace(type_obj)
    if namespace:
        return f'{namespace}.{type_obj.__name__}'
    return type_obj.__name__