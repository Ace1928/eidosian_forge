from suds import *
from logging import getLogger
def received(self, context):
    """
        Suds has received the specified reply.

        Provides the plugin with the opportunity to inspect/modify the received
        XML text before it is SAX parsed.

        @param context: The reply context.
            The I{reply} is the raw text.
        @type context: L{MessageContext}

        """
    pass