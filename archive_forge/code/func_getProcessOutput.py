import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def getProcessOutput(executable, args=(), env={}, path=None, reactor=None, errortoo=0):
    """
    Spawn a process and return its output as a deferred returning a L{bytes}.

    @param executable: The file name to run and get the output of - the
                       full path should be used.

    @param args: the command line arguments to pass to the process; a
                 sequence of strings. The first string should B{NOT} be the
                 executable's name.

    @param env: the environment variables to pass to the process; a
                dictionary of strings.

    @param path: the path to run the subprocess in - defaults to the
                 current directory.

    @param reactor: the reactor to use - defaults to the default reactor

    @param errortoo: If true, include stderr in the result.  If false, if
        stderr is received the returned L{Deferred} will errback with an
        L{IOError} instance with a C{processEnded} attribute.  The
        C{processEnded} attribute refers to a L{Deferred} which fires when the
        executed process ends.
    """
    return _callProtocolWithDeferred(lambda d: _BackRelay(d, errortoo=errortoo), executable, args, env, path, reactor)