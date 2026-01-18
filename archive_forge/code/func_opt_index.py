import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_index(self, indexName):
    """
        Add the name of a file used to check for directory indexes.
        [default: index, index.html]
        """
    self['indexes'].append(indexName)