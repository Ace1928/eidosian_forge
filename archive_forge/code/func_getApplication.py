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
def getApplication(config, passphrase):
    s = [(config[t], t) for t in ['python', 'source', 'file'] if config[t]][0]
    filename, style = (s[0], {'file': 'pickle'}.get(s[1], s[1]))
    try:
        log.msg('Loading %s...' % filename)
        application = service.loadApplication(filename, style, passphrase)
        log.msg('Loaded.')
    except Exception as e:
        s = 'Failed to load application: %s' % e
        if isinstance(e, KeyError) and e.args[0] == 'application':
            s += "\nCould not find 'application' in the file. To use 'twistd -y', your .tac\nfile must create a suitable object (e.g., by calling service.Application())\nand store it in a variable named 'application'. twistd loads your .tac file\nand scans the global variables for one of this name.\n\nPlease read the 'Using Application' HOWTO for details.\n"
        traceback.print_exc(file=log.logfile)
        log.msg(s)
        log.deferr()
        sys.exit('\n' + s + '\n')
    return application