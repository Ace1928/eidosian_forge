import random
from itertools import chain
from typing import Dict, List, Optional, Tuple
from twisted.conch import error
from twisted.conch.ssh import _kex, connection, transport, userauth
from twisted.internet import protocol
from twisted.logger import Logger
def getPublicKeys(self):
    """
        Called when the factory is started to get the public portions of the
        servers host keys.  Returns a dictionary mapping SSH key types to
        public key strings.

        @rtype: L{dict}
        """
    raise NotImplementedError('getPublicKeys unimplemented')