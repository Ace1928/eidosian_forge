import dataclasses
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
def is_dataclass_instance(obj: object) -> bool:
    """Check if object is dataclass."""
    return dataclasses.is_dataclass(obj) and (not isinstance(obj, type))