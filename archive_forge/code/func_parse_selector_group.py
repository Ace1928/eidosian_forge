import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
def parse_selector_group(stream: 'TokenStream') -> Iterator[Selector]:
    stream.skip_whitespace()
    while 1:
        yield Selector(*parse_selector(stream))
        if stream.peek() == ('DELIM', ','):
            stream.next()
            stream.skip_whitespace()
        else:
            break