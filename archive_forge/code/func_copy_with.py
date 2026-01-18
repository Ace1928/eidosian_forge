import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def copy_with(self, params):
    assert len(params) == 1
    new_type = params[0]
    return _AnnotatedAlias(new_type, self.__metadata__)