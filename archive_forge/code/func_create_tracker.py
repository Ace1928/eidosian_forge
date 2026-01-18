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
def create_tracker(fn, track_tqdm):
    progress = Progress(track_tqdm=track_tqdm)
    if not track_tqdm:
        return (progress, fn)
    return (progress, utils.function_wrapper(f=fn, before_fn=LocalContext.progress.set, before_args=(progress,), after_fn=LocalContext.progress.set, after_args=(None,)))