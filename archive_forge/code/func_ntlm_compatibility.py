import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
@ntlm_compatibility.setter
def ntlm_compatibility(self, value):
    self._context.ntlm_compatibility = value