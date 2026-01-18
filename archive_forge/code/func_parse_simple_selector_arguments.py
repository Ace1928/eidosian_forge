import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def parse_simple_selector_arguments(stream: 'TokenStream') -> List[Tree]:
    arguments = []
    while 1:
        result, pseudo_element = parse_simple_selector(stream, True)
        if pseudo_element:
            raise SelectorSyntaxError('Got pseudo-element ::%s inside function' % (pseudo_element,))
        stream.skip_whitespace()
        next = stream.next()
        if next in (('EOF', None), ('DELIM', ',')):
            stream.next()
            stream.skip_whitespace()
            arguments.append(result)
        elif next == ('DELIM', ')'):
            arguments.append(result)
            break
        else:
            raise SelectorSyntaxError('Expected an argument, got %s' % (next,))
    return arguments