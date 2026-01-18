from __future__ import annotations
import logging
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import cast
from sqlalchemy import schema
from sqlalchemy import text
from . import _autogen
from . import base
from ._autogen import _constraint_sig as _constraint_sig
from ._autogen import ComparisonResult as ComparisonResult
from .. import util
from ..util import sqla_compat
def check_dicts(meta_dict: Mapping[str, Any], insp_dict: Mapping[str, Any], default_dict: Mapping[str, Any], attrs: Iterable[str]):
    for attr in set(attrs).difference(skip):
        meta_value = meta_dict.get(attr)
        insp_value = insp_dict.get(attr)
        if insp_value != meta_value:
            default_value = default_dict.get(attr)
            if meta_value == default_value:
                ignored_attr.add(attr)
            else:
                diff.add(attr)