from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Any, Dict, Iterable, Set, FrozenSet, Optional
from interegular import InvalidSyntax, REFlags
from interegular.fsm import FSM, Alphabet, anything_else
from interegular.patterns import Pattern, Unsupported, parse_pattern
from interegular.utils import logger, soft_repr
def format_multiline(self, intro: str='Example Collision: ', indent: str='', force_pointer: bool=False) -> str:
    """
        Formats this example somewhat similar to a python syntax error.
        - intro is added on the first line
        - indent is added on the second line
        The three parts of the example are concatenated and `^` is used to underline them.

        ExampleCollision(prefix='a', main_text='cd', postfix='ef').format_multiline()

        leads to

        Example Collision: acdef
                             ^^

        This function will escape the character where necessary to stay readable.
        if `force_pointer` is False, the function will not produce the second line if only main_text is set
        """
    if len(intro) < len(indent):
        raise ValueError("Can't have intro be shorter than indent")
    prefix = soft_repr(self.prefix)
    main_text = soft_repr(self.main_text)
    postfix = soft_repr(self.postfix)
    text = f'{prefix}{main_text}{postfix}'
    if len(text) != len(main_text):
        whitespace = ' ' * (len(intro) - len(indent) + len(prefix))
        pointers = '^' * len(main_text)
        return f'{intro}{text}\n{indent}{whitespace}{pointers}'
    else:
        return f'{intro}{text}'