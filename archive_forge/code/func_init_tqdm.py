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
def init_tqdm(self, iterable=None, desc=None, total=None, unit='steps', *args, **kwargs):
    self._progress = LocalContext.progress.get()
    if self._progress is not None:
        self._progress.tqdm(iterable, desc, total, unit, _tqdm=self)
        kwargs['file'] = open(os.devnull, 'w')
    self.__init__orig__(iterable, desc, total, *args, unit=unit, **kwargs)