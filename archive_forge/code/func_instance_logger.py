from __future__ import annotations
import logging
import sys
from typing import Any
from typing import Optional
from typing import overload
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union
from .util import py311
from .util import py38
from .util.typing import Literal
def instance_logger(instance: Identified, echoflag: _EchoFlagType=None) -> None:
    """create a logger for an instance that implements :class:`.Identified`."""
    if instance.logging_name:
        name = '%s.%s' % (_qual_logger_name_for_cls(instance.__class__), instance.logging_name)
    else:
        name = _qual_logger_name_for_cls(instance.__class__)
    instance._echo = echoflag
    logger: Union[logging.Logger, InstanceLogger]
    if echoflag in (False, None):
        logger = logging.getLogger(name)
    else:
        logger = InstanceLogger(echoflag, name)
    instance.logger = logger