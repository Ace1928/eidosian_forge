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
def dataclass_json_dict(obj: Any) -> Dict[str, Any]:
    """Return a dictionary suitable for `_json_dict_` from a dataclass.

    Dataclasses keep track of their relevant fields, so we can automatically generate these.

    Dataclasses are implemented with somewhat complex metaprogramming, and tooling (PyCharm, mypy)
    have special cases for dealing with classes decorated with @dataclass. There is very little
    support (and no plans for support) for decorators that wrap @dataclass (like
    @cirq.json_serializable_dataclass) or combining additional decorators with @dataclass.
    Although not as elegant, you may want to consider explicitly defining `_json_dict_` on your
    dataclasses which simply `return dataclass_json_dict(self)`.
    """
    attribute_names = [f.name for f in dataclasses.fields(obj)]
    return obj_to_dict_helper(obj, attribute_names)