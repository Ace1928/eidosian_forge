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
def sync_lazy_cache(self, example_index, *input_values):
    cached_index = self._get_cached_index_if_cached(example_index)
    if cached_index is not None:
        output = self.load_from_cache(cached_index)
        yield (output[0] if len(self.outputs) == 1 else output)
        return
    output = [None] * len(self.outputs)
    if inspect.isgeneratorfunction(self.fn):
        fn = self.fn
    else:
        fn = utils.sync_fn_to_generator(self.fn)
    for output in fn(*input_values):
        output = client_utils.synchronize_async(self._postprocess_output, output)
        yield (output[0] if len(self.outputs) == 1 else output)
    self.cache_logger.flag(output)
    with open(self.cached_indices_file, 'a') as f:
        f.write(f'{example_index}\n')