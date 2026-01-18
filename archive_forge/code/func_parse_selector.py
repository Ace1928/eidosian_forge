import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def parse_selector(stream: 'TokenStream') -> Tuple[Tree, Optional[PseudoElement]]:
    result, pseudo_element = parse_simple_selector(stream)
    while 1:
        stream.skip_whitespace()
        peek = stream.peek()
        if peek in (('EOF', None), ('DELIM', ',')):
            break
        if pseudo_element:
            raise SelectorSyntaxError('Got pseudo-element ::%s not at the end of a selector' % pseudo_element)
        if peek.is_delim('+', '>', '~'):
            combinator = typing.cast(str, stream.next().value)
            stream.skip_whitespace()
        else:
            combinator = ' '
        next_selector, pseudo_element = parse_simple_selector(stream)
        result = CombinedSelector(result, combinator, next_selector)
    return (result, pseudo_element)