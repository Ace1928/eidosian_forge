import functools
import os
import typing
def open_pkg_binary(path: str) -> typing.BinaryIO:
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path), 'rb')