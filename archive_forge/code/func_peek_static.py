from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
def peek_static(self, expected: str) -> bool:
    l = len(expected)
    if self.data[self.index:self.index + l] == expected:
        return True
    else:
        self._expected[self.index].append(expected)
        return False