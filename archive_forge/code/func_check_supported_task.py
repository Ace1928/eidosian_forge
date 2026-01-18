import platform
from dataclasses import field
from enum import Enum
from typing import Dict, List, Optional, Union
from . import is_pydantic_available
from .doc import generate_doc_dataclass
@staticmethod
def check_supported_task(task: str):
    if task not in APIFeaturesManager._SUPPORTED_TASKS:
        raise KeyError(f'{task} is not supported yet. Only {APIFeaturesManager._SUPPORTED_TASKS} are supported. If you want to support {task} please propose a PR or open up an issue.')