from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def is_namespace_extension(obj: Any) -> bool:
    return isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str)