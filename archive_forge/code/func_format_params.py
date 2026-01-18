import email.utils
import typing
from datetime import datetime, timezone
from decimal import Decimal
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing_extensions import Protocol, TypeGuard
from . import Consts
from .GithubException import BadAttributeException, IncompletableObject
def format_params(params: Dict[str, Any]) -> typing.Generator[str, None, None]:
    items = list(params.items())
    for k, v in sorted(items, key=itemgetter(0), reverse=True):
        if isinstance(v, bytes):
            v = v.decode('utf-8')
        if isinstance(v, str):
            v = f'"{v}"'
        yield f'{k}={v}'