from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
def render_yaml(self, objs: Union[Any, List[Any]], level: str='INFO', **kwargs) -> None:
    """
        Tries to render a yaml object.
        """
    _log = self.get_log_mode(level)
    try:
        from fileio.io import Yaml
        _log('\n' + Yaml.dumps(objs, **kwargs))
    except Exception as e:
        import yaml
        _log('\n' + yaml.dump(objs, **kwargs))