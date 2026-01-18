import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
def remove_none_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}