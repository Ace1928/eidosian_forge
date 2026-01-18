import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
def copy_dict(*, src: Dict[str, Any], dest: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            for dict_k, dict_v in v.items():
                dest[f'{k}[{dict_k}]'] = dict_v
        else:
            dest[k] = v