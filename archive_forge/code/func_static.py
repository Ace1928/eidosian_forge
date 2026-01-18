from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
def static(self, expected: str):
    length = len(expected)
    if self.data[self.index:self.index + length] == expected:
        self.index += length
    else:
        self._expected[self.index].append(expected)
        raise nomatch