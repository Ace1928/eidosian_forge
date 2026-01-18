from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.core import log
def has_unparsed(self, message_class):
    """Checks if the set contains an unparsed message of the given type.

    This differs from has() when the set contains a message of the given type
    with a parse error.  has() will return false when this is the case, but
    has_unparsed() will return true.  This is only useful for error checking.
    """
    return message_class.MESSAGE_TYPE_ID in self.items