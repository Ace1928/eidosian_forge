import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_listen(self, port):
    """
        Add an strports description of port to start the server on.
        [default: tcp:8080]
        """
    self['ports'].append(port)