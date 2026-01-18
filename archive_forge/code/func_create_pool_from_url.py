from __future__ import annotations
import inspect
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Type
from typing import Union
from . import base
from . import url as _url
from .interfaces import DBAPIConnection
from .mock import create_mock_engine
from .. import event
from .. import exc
from .. import util
from ..pool import _AdhocProxiedConnection
from ..pool import ConnectionPoolEntry
from ..sql import compiler
from ..util import immutabledict
def create_pool_from_url(url: Union[str, URL], **kwargs: Any) -> Pool:
    """Create a pool instance from the given url.

    If ``poolclass`` is not provided the pool class used
    is selected using the dialect specified in the URL.

    The arguments passed to :func:`_sa.create_pool_from_url` are
    identical to the pool argument passed to the :func:`_sa.create_engine`
    function.

    .. versionadded:: 2.0.10
    """
    for key in _pool_translate_kwargs:
        if key in kwargs:
            kwargs[_pool_translate_kwargs[key]] = kwargs.pop(key)
    engine = create_engine(url, **kwargs, _initialize=False)
    return engine.pool