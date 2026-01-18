import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_ignore_ext(self, ext):
    """
        Specify an extension to ignore.  These will be processed in order.
        """
    if not isinstance(self['root'], static.File):
        raise usage.UsageError('You can only use --ignore_ext after --path.')
    self['root'].ignoreExt(ext)