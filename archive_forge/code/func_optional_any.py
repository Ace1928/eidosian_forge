import collections.abc
import io
import itertools
import types
import typing
def optional_any(elements) -> typing.Optional[bool]:
    if any(elements):
        return True
    if any((e is None for e in elements)):
        return unknown
    return False