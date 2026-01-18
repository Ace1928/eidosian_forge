import abc
import io
import os
from typing import Any, BinaryIO, Iterable, Iterator, NoReturn, Text, Optional
from typing import runtime_checkable, Protocol
from typing import Union
def open_resource(self, resource: StrPath) -> io.BufferedReader:
    return self.files().joinpath(resource).open('rb')