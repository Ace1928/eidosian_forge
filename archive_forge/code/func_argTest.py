from __future__ import annotations
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
def argTest(self, argKlass: Callable[Concatenate[str, _P], formmethod.Argument], testPairs: Iterable[tuple[object, object]], badValues: Iterable[object], *args: _P.args, **kwargs: _P.kwargs) -> None:
    arg = argKlass('name', *args, **kwargs)
    for val, result in testPairs:
        self.assertEqual(arg.coerce(val), result)
    for val in badValues:
        self.assertRaises(formmethod.InputError, arg.coerce, val)