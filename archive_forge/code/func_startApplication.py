import getpass
import os
import pdb
import signal
import sys
import traceback
import warnings
from operator import attrgetter
from twisted import copyright, logger, plugin
from twisted.application import reactors, service
from twisted.application.reactors import NoSuchReactor, installReactor
from twisted.internet import defer
from twisted.internet.interfaces import _ISupportsExitSignalCapturing
from twisted.persisted import sob
from twisted.python import failure, log, logfile, runtime, usage, util
from twisted.python.reflect import namedAny, namedModule, qual
def startApplication(application, save):
    from twisted.internet import reactor
    service.IService(application).startService()
    if save:
        p = sob.IPersistable(application)
        reactor.addSystemEventTrigger('after', 'shutdown', p.save, 'shutdown')
    reactor.addSystemEventTrigger('before', 'shutdown', service.IService(application).stopService)