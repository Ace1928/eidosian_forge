from __future__ import annotations
import os
import contextlib
from enum import Enum
from pathlib import Path
from pydantic import model_validator
from lazyops.utils.logs import Logger, null_logger
from lazyops.imports._pydantic import BaseAppSettings, BaseModel
from lazyops.libs.abcs.state import GlobalContext
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from typing import List, Optional, Dict, Any, Callable, Union, Type, TYPE_CHECKING
from .types import AppEnv, get_app_env
def set_app_env(self, env: AppEnv) -> None:
    """
        Sets the app environment
        """
    self.app_env = self.app_env.from_env(env)