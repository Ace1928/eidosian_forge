from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from types import FunctionType, MethodType
from typing import Generic, TypeVar, Optional, List
def any_but(self, *strings, length: int=1) -> str:
    if self.index + length <= len(self.data):
        res = self.data[self.index:self.index + length]
        if res not in strings:
            self.index += length
            return res
        else:
            self._expected[self.index].append(f'<Any {length} except {strings}>')
            raise nomatch
    else:
        self._expected[self.index].append(f'<Any {length} except {strings}>')
        raise nomatch