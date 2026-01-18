import os
import warnings
import incremental
from twisted.application import service, strports
from twisted.internet import interfaces, reactor
from twisted.python import deprecate, reflect, threadpool, usage
from twisted.spread import pb
from twisted.web import demo, distrib, resource, script, server, static, twcgi, wsgi
def opt_resource_script(self, name):
    """
        An .rpy file to be used as the root resource of the webserver.
        """
    self['root'] = script.ResourceScriptWrapper(name)