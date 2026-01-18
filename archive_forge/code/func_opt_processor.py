import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_processor(self, proc):
    """
        `ext=class' where `class' is added as a Processor for files ending
        with `ext'.
        """
    if not isinstance(self['root'], static.File):
        raise usage.UsageError('You can only use --processor after --path.')
    ext, klass = proc.split('=', 1)
    self['root'].processors[ext] = reflect.namedClass(klass)