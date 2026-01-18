from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
def is_data_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and 'type' in value and (value['type'] == DATA_LIB_KEY) and ('data' in value)