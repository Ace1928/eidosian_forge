import random
from itertools import chain
from typing import Dict, List, Optional, Tuple
from twisted.conch import error
from twisted.conch.ssh import _kex, connection, transport, userauth
from twisted.internet import protocol
from twisted.logger import Logger
def getPrivateKeys(self):
    """
        Called when the factory is started to get the  private portions of the
        servers host keys.  Returns a dictionary mapping SSH key types to
        L{twisted.conch.ssh.keys.Key} objects.

        @rtype: L{dict}
        """
    raise NotImplementedError('getPrivateKeys unimplemented')