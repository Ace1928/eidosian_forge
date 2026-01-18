import abc
import mimetypes
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Generic, Callable, Iterable, Optional, Tuple, TypeVar, Union
from .cloudpath import CloudImplementation, CloudPath, implementation_registry
from .enums import FileCacheMode
from .exceptions import InvalidConfigurationException
def set_as_default_client(self) -> None:
    """Set this client instance as the default one used when instantiating cloud path
        instances for this cloud without a client specified."""
    self.__class__._default_client = self