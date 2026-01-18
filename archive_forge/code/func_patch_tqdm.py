from __future__ import annotations
import ast
import csv
import inspect
import os
import shutil
import subprocess
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional
import numpy as np
import PIL
import PIL.Image
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import components, oauth, processing_utils, routes, utils, wasm_utils
from gradio.context import Context, LocalContext
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import EventData
from gradio.exceptions import Error
from gradio.flagging import CSVLogger
def patch_tqdm() -> None:
    try:
        _tqdm = __import__('tqdm')
    except ModuleNotFoundError:
        return

    def init_tqdm(self, iterable=None, desc=None, total=None, unit='steps', *args, **kwargs):
        self._progress = LocalContext.progress.get()
        if self._progress is not None:
            self._progress.tqdm(iterable, desc, total, unit, _tqdm=self)
            kwargs['file'] = open(os.devnull, 'w')
        self.__init__orig__(iterable, desc, total, *args, unit=unit, **kwargs)

    def iter_tqdm(self):
        if self._progress is not None:
            return self._progress
        return self.__iter__orig__()

    def update_tqdm(self, n=1):
        if self._progress is not None:
            self._progress.update(n)
        return self.__update__orig__(n)

    def close_tqdm(self):
        if self._progress is not None:
            self._progress.close(self)
        return self.__close__orig__()

    def exit_tqdm(self, exc_type, exc_value, traceback):
        if self._progress is not None:
            self._progress.close(self)
        return self.__exit__orig__(exc_type, exc_value, traceback)
    if not hasattr(_tqdm.tqdm, '__init__orig__'):
        _tqdm.tqdm.__init__orig__ = _tqdm.tqdm.__init__
    if not hasattr(_tqdm.tqdm, '__update__orig__'):
        _tqdm.tqdm.__update__orig__ = _tqdm.tqdm.update
    if not hasattr(_tqdm.tqdm, '__close__orig__'):
        _tqdm.tqdm.__close__orig__ = _tqdm.tqdm.close
    if not hasattr(_tqdm.tqdm, '__exit__orig__'):
        _tqdm.tqdm.__exit__orig__ = _tqdm.tqdm.__exit__
    if not hasattr(_tqdm.tqdm, '__iter__orig__'):
        _tqdm.tqdm.__iter__orig__ = _tqdm.tqdm.__iter__
    _tqdm.tqdm.__init__ = init_tqdm
    _tqdm.tqdm.update = update_tqdm
    _tqdm.tqdm.close = close_tqdm
    _tqdm.tqdm.__exit__ = exit_tqdm
    _tqdm.tqdm.__iter__ = iter_tqdm
    if hasattr(_tqdm, 'auto') and hasattr(_tqdm.auto, 'tqdm'):
        _tqdm.auto.tqdm = _tqdm.tqdm