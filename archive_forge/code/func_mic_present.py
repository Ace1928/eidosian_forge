import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
@property
def mic_present(self):
    if self._authenticate_message:
        return bool(self._authenticate_message.mic)
    return False