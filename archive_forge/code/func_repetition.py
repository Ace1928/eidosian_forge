from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def repetition(self, base: _Repeatable):
    if self.static_b('*'):
        if self.static_b('?'):
            pass
        return _Repeated(base, 0, None)
    elif self.static_b('+'):
        if self.static_b('?'):
            pass
        return _Repeated(base, 1, None)
    elif self.static_b('?'):
        if self.static_b('?'):
            pass
        return _Repeated(base, 0, 1)
    elif self.static_b('{'):
        try:
            n = self.number()
        except nomatch:
            n = 0
        if self.static_b(','):
            try:
                m = self.number()
            except nomatch:
                m = None
        else:
            m = n
        self.static('}')
        if self.static_b('?'):
            pass
        return _Repeated(base, n, m)
    else:
        return base