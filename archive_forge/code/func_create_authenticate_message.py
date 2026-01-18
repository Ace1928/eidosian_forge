import base64
import struct
from ntlm_auth.constants import NegotiateFlags
from ntlm_auth.exceptions import NoAuthContextError
from ntlm_auth.messages import AuthenticateMessage, ChallengeMessage, \
from ntlm_auth.session_security import SessionSecurity
def create_authenticate_message(self, user_name, password, domain_name=None, workstation=None, server_certificate_hash=None):
    self._context.username = user_name
    self._context.password = password
    self._context.domain = domain_name
    self._context.workstation = workstation
    self._context._server_certificate_hash = server_certificate_hash
    msg = self._context.step(self._challenge_token)
    return base64.b64encode(msg)