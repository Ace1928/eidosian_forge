import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.load.dump import dumpd
from langchain_core.memory import BaseMemory
from langchain_core.outputs import RunInfo
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from langchain_core.runnables import (
from langchain_core.runnables.utils import create_model
from langchain.schema import RUN_KEY
@root_validator()
def raise_callback_manager_deprecation(cls, values: Dict) -> Dict:
    """Raise deprecation warning if callback_manager is used."""
    if values.get('callback_manager') is not None:
        if values.get('callbacks') is not None:
            raise ValueError('Cannot specify both callback_manager and callbacks. callback_manager is deprecated, callbacks is the preferred parameter to pass in.')
        warnings.warn('callback_manager is deprecated. Please use callbacks instead.', DeprecationWarning)
        values['callbacks'] = values.pop('callback_manager', None)
    return values