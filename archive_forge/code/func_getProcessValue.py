import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def getProcessValue(executable, args=(), env={}, path=None, reactor=None):
    """Spawn a process and return its exit code as a Deferred."""
    return _callProtocolWithDeferred(_ValueGetter, executable, args, env, path, reactor)