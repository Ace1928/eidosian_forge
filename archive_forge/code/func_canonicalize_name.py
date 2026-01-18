import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
def canonicalize_name(name: str) -> NormalizedName:
    value = _canonicalize_regex.sub('-', name).lower()
    return cast(NormalizedName, value)