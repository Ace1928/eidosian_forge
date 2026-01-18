from __future__ import annotations
import logging # isort:skip
import os
from types import ModuleType
from ...core.types import PathLike
from ...util.callback_manager import _check_callback
from .code_runner import CodeRunner
from .request_handler import RequestHandler
def extract_callbacks() -> None:
    contents = self._module.__dict__
    if 'process_request' in contents:
        self._process_request = contents['process_request']
    _check_callback(self._process_request, ('request',), what='process_request')