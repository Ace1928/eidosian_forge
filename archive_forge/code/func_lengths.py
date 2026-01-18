from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@property
def lengths(self) -> Tuple[int, Optional[int]]:
    """Returns the minimum and maximum length that this pattern can match
         (maximum can be None bei infinite length)"""
    if not hasattr(self, '_lengths_cache'):
        super(_BasePattern, self).__setattr__('_lengths_cache', self._get_lengths())
    return self._lengths_cache