import abc
import io
import os
from typing import Any, BinaryIO, Iterable, Iterator, NoReturn, Text, Optional
from typing import runtime_checkable, Protocol
from typing import Union
def resource_path(self, resource: Any) -> NoReturn:
    raise FileNotFoundError(resource)