from __future__ import annotations
import contextlib
import mimetypes
from abc import ABC, abstractmethod
from io import BufferedReader, BytesIO
from pathlib import PurePath
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Union, cast
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
@contextlib.contextmanager
def as_bytes_io(self) -> Generator[Union[BytesIO, BufferedReader], None, None]:
    """Read data as a byte stream."""
    if isinstance(self.data, bytes):
        yield BytesIO(self.data)
    elif self.data is None and self.path:
        with open(str(self.path), 'rb') as f:
            yield f
    else:
        raise NotImplementedError(f'Unable to convert blob {self}')