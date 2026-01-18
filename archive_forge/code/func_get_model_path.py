from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import pathlib
import copy
import re
from typing import Dict, Iterable, List, Tuple, Union, Type, Callable
from utils.log import quick_log
from fastapi import HTTPException
from pydantic import BaseModel, Field
from routes import state_cache
import global_var
def get_model_path(model_path: str) -> str:
    if os.path.isabs(model_path):
        return model_path
    working_dir: pathlib.Path = pathlib.Path(os.path.abspath(os.getcwd()))
    parent_paths: List[pathlib.Path] = [working_dir, working_dir.parent, pathlib.Path(os.path.abspath(__file__)).parent.parent, pathlib.Path(os.path.abspath(__file__)).parent.parent.parent]
    child_paths: List[Callable[[pathlib.Path], pathlib.Path]] = [lambda p: p / model_path, lambda p: p / 'build' / 'bin' / model_path]
    for parent_path in parent_paths:
        for child_path in child_paths:
            full_path: pathlib.Path = child_path(parent_path)
            if os.path.isfile(full_path):
                return str(full_path)
    return model_path