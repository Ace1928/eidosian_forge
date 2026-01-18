import inspect
import os
import shutil
import warnings
from pathlib import Path, PurePath
from typing import Dict, List, Mapping, Optional, Sequence, Union
import huggingface_hub
from . import config
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager
from .info import DatasetInfo
from .load import (
from .utils.deprecation_utils import deprecated
from .utils.file_utils import relative_to_absolute_path
from .utils.logging import get_logger
from .utils.version import Version
@deprecated("Use 'evaluate.inspect_evaluation_module' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate")
def inspect_metric(path: str, local_path: str, download_config: Optional[DownloadConfig]=None, **download_kwargs):
    """
    Allow inspection/modification of a metric script by copying it on local drive at local_path.

    <Deprecated version="2.5.0">

    Use `evaluate.inspect_evaluation_module` instead, from the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>

    Args:
        path (``str``): path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with ``datasets.list_datasets()``)
                e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``
        local_path (``str``): path to the local folder to copy the datset script to.
        download_config (Optional ``datasets.DownloadConfig``): specific download configuration parameters.
        **download_kwargs (additional keyword arguments): optional attributes for DownloadConfig() which will override the attributes in download_config if supplied.
    """
    metric_module = metric_module_factory(path, download_config=download_config, **download_kwargs)
    metric_cls = import_main_class(metric_module.module_path, dataset=False)
    module_source_path = inspect.getsourcefile(metric_cls)
    module_source_dirpath = os.path.dirname(module_source_path)
    for dirpath, dirnames, filenames in os.walk(module_source_dirpath):
        dst_dirpath = os.path.join(local_path, os.path.relpath(dirpath, module_source_dirpath))
        os.makedirs(dst_dirpath, exist_ok=True)
        dirnames[:] = [dirname for dirname in dirnames if not dirname.startswith(('.', '__'))]
        for filename in filenames:
            shutil.copy2(os.path.join(dirpath, filename), os.path.join(dst_dirpath, filename))
        shutil.copystat(dirpath, dst_dirpath)
    local_path = relative_to_absolute_path(local_path)
    print(f'The processing scripts for metric {path} can be inspected at {local_path}. The main class is in {module_source_dirpath}. You can modify this processing scripts and use it with `datasets.load_metric("{PurePath(local_path).as_posix()}")`.')