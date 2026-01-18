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
def makeStatBar(width, maxPosition, doneChar='=', undoneChar='-', currentChar='>'):
    """
    Creates a function that will return a string representing a progress bar.
    """
    aValue = width / float(maxPosition)

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
    statBar.__doc__ = "statBar(position, force = 0) -> '[%s%s%s]'-style progress bar\n\n    returned string is %d characters long, and the range goes from 0..%d.\n    The 'position' argument is where the '%s' will be drawn.  If force is false,\n    '' will be returned instead if the resulting progress bar is identical to the\n    previously returned progress bar.\n" % (doneChar * 3, currentChar, undoneChar * 3, width, maxPosition, currentChar)
    return statBar