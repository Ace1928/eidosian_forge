from typing import Dict, Tuple, Union
from twisted.words.protocols.jabber.xmpp_stringprep import (
def internJID(jidstring):
    """
    Return interned JID.

    @rtype: L{JID}
    """
    if jidstring in __internJIDs:
        return __internJIDs[jidstring]
    else:
        j = JID(jidstring)
        __internJIDs[jidstring] = j
        return j