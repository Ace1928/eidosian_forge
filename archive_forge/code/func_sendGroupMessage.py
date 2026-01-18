from zope.interface import Attribute, Interface
def sendGroupMessage(text, metadata=None):
    """
        Send a message to this group.

        @type text: str

        @type metadata: dict
        @param metadata: Valid keys for this dictionary include:

            - C{'style'}: associated with one of:
                - C{'emote'}: indicates this is an action
        """