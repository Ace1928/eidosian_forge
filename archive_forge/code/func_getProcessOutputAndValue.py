import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def getProcessOutputAndValue(executable, args=(), env={}, path=None, reactor=None, stdinBytes=None):
    """Spawn a process and returns a Deferred that will be called back with
    its output (from stdout and stderr) and it's exit code as (out, err, code)
    If a signal is raised, the Deferred will errback with the stdout and
    stderr up to that point, along with the signal, as (out, err, signalNum)
    """
    return _callProtocolWithDeferred(_EverythingGetter, executable, args, env, path, reactor, protoArgs=(stdinBytes,))