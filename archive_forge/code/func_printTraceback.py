import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def printTraceback(self, file=None, elideFrameworkCode=False, detail='default'):
    """
        Emulate Python's standard error reporting mechanism.

        @param file: If specified, a file-like object to which to write the
            traceback.

        @param elideFrameworkCode: A flag indicating whether to attempt to
            remove uninteresting frames from within Twisted itself from the
            output.

        @param detail: A string indicating how much information to include
            in the traceback.  Must be one of C{'brief'}, C{'default'}, or
            C{'verbose'}.
        """
    if file is None:
        from twisted.python import log
        file = log.logerr
    w = file.write
    if detail == 'verbose' and (not self.captureVars):
        formatDetail = 'verbose-vars-not-captured'
    else:
        formatDetail = detail
    if detail == 'verbose':
        w('*--- Failure #%d%s---\n' % (self.count, self.pickled and ' (pickled) ' or ' '))
    elif detail == 'brief':
        if self.frames:
            hasFrames = 'Traceback'
        else:
            hasFrames = 'Traceback (failure with no frames)'
        w('%s: %s: %s\n' % (hasFrames, reflect.safe_str(self.type), reflect.safe_str(self.value)))
    else:
        w('Traceback (most recent call last):\n')
    if self.frames:
        if not elideFrameworkCode:
            format_frames(self.stack[-traceupLength:], w, formatDetail)
            w(f'{EXCEPTION_CAUGHT_HERE}\n')
        format_frames(self.frames, w, formatDetail)
    elif not detail == 'brief':
        w('Failure: ')
    if not detail == 'brief':
        w(f'{reflect.qual(self.type)}: {reflect.safe_str(self.value)}\n')
    if isinstance(self.value, Failure):
        file.write(' (chained Failure)\n')
        self.value.printTraceback(file, elideFrameworkCode, detail)
    if detail == 'verbose':
        w('*--- End of Failure #%d ---\n' % self.count)