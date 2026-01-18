import abc
import io
import os
from typing import Any, BinaryIO, Iterable, Iterator, NoReturn, Text, Optional
from typing import runtime_checkable, Protocol
from typing import Union
def is_resource(self, path: StrPath) -> bool:
    return self.files().joinpath(path).is_file()