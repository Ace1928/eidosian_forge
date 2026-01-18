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
def class_logger(cls: Type[_IT]) -> Type[_IT]:
    logger = logging.getLogger(_qual_logger_name_for_cls(cls))
    cls._should_log_debug = lambda self: logger.isEnabledFor(logging.DEBUG)
    cls._should_log_info = lambda self: logger.isEnabledFor(logging.INFO)
    cls.logger = logger
    _logged_classes.add(cls)
    return cls