import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
def reset_rc4_state(self, outgoing=True):
    """ Resets the signing cipher for the incoming or outgoing cipher. For SPNEGO for calculating mechListMIC. """
    if self._session_security:
        self._session_security.reset_rc4_state(outgoing=outgoing)