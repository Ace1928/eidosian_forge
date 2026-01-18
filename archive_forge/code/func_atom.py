from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def atom(self):
    if self.static_b('['):
        return self.repetition(self.chargroup())
    elif self.static_b('\\'):
        return self.repetition(self.escaped())
    elif self.static_b('.'):
        return self.repetition(_DOT)
    elif self.static_b('$'):
        raise Unsupported("'$'")
    elif self.static_b('^'):
        raise Unsupported("'^'")
    else:
        c = self.any_but(*self.SPECIAL_CHARS_STANDARD)
        return self.repetition(_CharGroup(frozenset({c}), False))