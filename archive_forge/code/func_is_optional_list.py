import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def is_optional_list(v: Any, type: Union[Type, Tuple[Type, ...]]) -> bool:
    return isinstance(v, _NotSetType) or (isinstance(v, list) and all((isinstance(element, type) for element in v)))