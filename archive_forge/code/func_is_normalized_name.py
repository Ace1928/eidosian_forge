import re
from typing import FrozenSet, NewType, Tuple, Union, cast
from .tags import Tag, parse_tag
from .version import InvalidVersion, Version
def is_normalized_name(name: str) -> bool:
    return _normalized_regex.match(name) is not None