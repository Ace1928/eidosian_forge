from abc import ABC, abstractmethod
from typing import List, Iterator
from .exceptions import LarkError
from .lark import PostLex
from .lexer import Token
@property
def always_accept(self):
    return (self.NL_type,)