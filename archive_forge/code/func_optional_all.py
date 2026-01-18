import collections.abc
import io
import itertools
import types
import typing
def optional_all(elements) -> typing.Optional[bool]:
    if all(elements):
        return True
    if all((e is False for e in elements)):
        return False
    return unknown