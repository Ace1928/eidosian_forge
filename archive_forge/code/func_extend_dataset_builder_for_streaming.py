import importlib
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Optional
from .download.download_config import DownloadConfig
from .download.streaming_download_manager import (
from .utils.logging import get_logger
from .utils.patching import patch_submodule
from .utils.py_utils import get_imports
def extend_dataset_builder_for_streaming(builder: 'DatasetBuilder'):
    """Extend the dataset builder module and the modules imported by it to support streaming.

    Args:
        builder (:class:`DatasetBuilder`): Dataset builder instance.
    """
    download_config = DownloadConfig(storage_options=builder.storage_options, token=builder.token)
    extend_module_for_streaming(builder.__module__, download_config=download_config)
    if not builder.__module__.startswith('datasets.'):
        for imports in get_imports(inspect.getfile(builder.__class__)):
            if imports[0] == 'internal':
                internal_import_name = imports[1]
                internal_module_name = '.'.join(builder.__module__.split('.')[:-1] + [internal_import_name])
                extend_module_for_streaming(internal_module_name, download_config=download_config)
    from .builder import DatasetBuilder
    parent_builder_modules = [cls.__module__ for cls in type(builder).__mro__[1:] if issubclass(cls, DatasetBuilder) and cls.__module__ != DatasetBuilder.__module__]
    for module in parent_builder_modules:
        extend_module_for_streaming(module, download_config=download_config)