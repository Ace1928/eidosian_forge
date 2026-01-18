import types
import weakref
import six
from apitools.base.protorpclite import util
@property
def message_type(self):
    """Underlying message type used for serialization.

        Will always be a sub-class of Message.  This is different from type
        which represents the python value that message_type is mapped to for
        use by the user.
        """
    return self.type