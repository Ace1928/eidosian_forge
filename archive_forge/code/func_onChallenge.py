import re
from base64 import b64decode, b64encode
from twisted.internet import defer
from twisted.words.protocols.jabber import sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def onChallenge(self, element):
    """
        Parse challenge and send response from the mechanism.

        @param element: the challenge protocol element.
        @type element: L{domish.Element}.
        """
    try:
        challenge = fromBase64(str(element))
    except SASLIncorrectEncodingError:
        self._deferred.errback()
    else:
        self.sendResponse(self.mechanism.getResponse(challenge))