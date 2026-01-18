from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Flag, auto
from textwrap import indent
from typing import Iterable, FrozenSet, Optional, Tuple, Union
from interegular.fsm import FSM, anything_else, epsilon, Alphabet
from interegular.utils.simple_parser import SimpleParser, nomatch, NoMatch
def extension_group(self):
    c = self.any()
    if c in 'aiLmsux-':
        self.index -= 1
        added_flags = self.multiple('aiLmsux', 0, None)
        if self.static_b('-'):
            removed_flags = self.multiple('aiLmsux', 1, None)
        else:
            removed_flags = ''
        if self.static_b(':'):
            p = self.pattern()
            p = p.with_flags(_get_flags(added_flags), _get_flags(removed_flags))
            self.static(')')
            return self.repetition(p)
        elif removed_flags != '':
            raise nomatch
        else:
            self.static(')')
            self.flags = _get_flags(added_flags)
            return _EMPTY
    elif c == ':':
        p = self.pattern()
        self.static(')')
        return self.repetition(p)
    elif c == 'P':
        if self.static_b('<'):
            self.multiple('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_', 1, None)
            self.static('>')
            p = self.pattern()
            self.static(')')
            return self.repetition(p)
        elif self.static_b('='):
            raise Unsupported('Group references are not implemented')
    elif c == '#':
        while not self.static_b(')'):
            self.any()
    elif c == '=':
        p = self.pattern()
        self.static(')')
        return _NonCapturing(p, False, False)
    elif c == '!':
        p = self.pattern()
        self.static(')')
        return _NonCapturing(p, False, True)
    elif c == '<':
        c = self.any()
        if c == '=':
            p = self.pattern()
            self.static(')')
            return _NonCapturing(p, True, False)
        elif c == '!':
            p = self.pattern()
            self.static(')')
            return _NonCapturing(p, True, True)
    elif c == '(':
        raise Unsupported('Conditional matching is not implemented')
    else:
        raise InvalidSyntax(f'Unknown group-extension: {c!r} (Context: {self.data[self.index - 3:self.index + 5]!r}')