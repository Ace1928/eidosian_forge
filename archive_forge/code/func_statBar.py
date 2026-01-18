from __future__ import annotations
import errno
import os
import sys
import warnings
from typing import AnyStr
from collections import OrderedDict
from typing import (
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def statBar(position, force=0, last=['']):
    assert len(last) == 1, "Don't mess with the last parameter."
    done = int(aValue * position)
    toDo = width - done - 2
    result = f'[{doneChar * done}{currentChar}{undoneChar * toDo}]'
    if force:
        last[0] = result
        return result
    if result == last[0]:
        return ''
    last[0] = result
    return result