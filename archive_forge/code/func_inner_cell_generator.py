import os
import sys
import traceback
from typing import Generator, Tuple, Union, List
import nbformat
import tornado.web
from jupyter_server.config_manager import recursive_update
from nbclient.exceptions import CellExecutionError
from nbclient.util import ensure_async
from nbconvert.preprocessors import ClearOutputPreprocessor
from traitlets.config.configurable import LoggingConfigurable
from .execute import VoilaExecutor, strip_code_cell_warnings
from .exporter import VoilaExporter
from .paths import collect_template_paths
from .utils import ENV_VARIABLE
def inner_cell_generator(nb, kernel_id):
    return self._jinja_cell_generator(nb, kernel_id)