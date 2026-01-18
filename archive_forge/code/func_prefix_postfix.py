from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
@property
def prefix_postfix(self) -> Tuple[int, Optional[int]]:
    """Returns the number of dots that have to be pre-/postfixed to support look(aheads|backs)"""
    if not hasattr(self, '_prefix_cache'):
        super(_BasePattern, self).__setattr__('_prefix_cache', self._get_prefix_postfix())
    return self._prefix_cache