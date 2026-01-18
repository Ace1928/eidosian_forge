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
def fixPdb():

    def do_stop(self, arg):
        self.clear_all_breaks()
        self.set_continue()
        from twisted.internet import reactor
        reactor.callLater(0, reactor.stop)
        return 1

    def help_stop(self):
        print('stop - Continue execution, then cleanly shutdown the twisted reactor.')

    def set_quit(self):
        os._exit(0)
    pdb.Pdb.set_quit = set_quit
    pdb.Pdb.do_stop = do_stop
    pdb.Pdb.help_stop = help_stop