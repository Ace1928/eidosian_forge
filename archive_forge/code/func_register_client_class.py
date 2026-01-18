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
def register_client_class(key: str) -> Callable:

    def decorator(cls: type) -> type:
        if not issubclass(cls, Client):
            raise TypeError('Only subclasses of Client can be registered.')
        implementation_registry[key]._client_class = cls
        implementation_registry[key].name = key
        cls._cloud_meta = implementation_registry[key]
        return cls
    return decorator