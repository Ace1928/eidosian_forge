import base64
import calendar
import hashlib
import hmac
import os
import struct
import time
import ntlm_auth.compute_hash as comphash
import ntlm_auth.compute_keys as compkeys
import ntlm_auth.messages
from ntlm_auth.des import DES
from ntlm_auth.constants import AvId, AvFlags, NegotiateFlags
from ntlm_auth.gss_channel_bindings import GssChannelBindingsStruct
def get_lm_challenge_response(self):
    """
        [MS-NLMP] v28.0 2016-07-14

        3.3.1 - NTLM v1 Authentication
        3.3.2 - NTLM v2 Authentication

        This method returns the LmChallengeResponse key based on the
        ntlm_compatibility chosen and the target_info supplied by the
        CHALLENGE_MESSAGE. It is quite different from what is set in the
        document as it combines the NTLMv1, NTLM2 and NTLMv2 methods into one
        and calls separate methods based on the ntlm_compatibility flag chosen.

        :return: response (LmChallengeResponse) - The LM response to the server
            challenge. Computed by the client
        """
    if self._negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY and self._ntlm_compatibility < 3:
        response = self._get_LMv1_with_session_security_response(self._client_challenge)
    elif 0 <= self._ntlm_compatibility <= 1:
        response = self._get_LMv1_response(self._password, self._server_challenge)
    elif self._ntlm_compatibility == 2:
        response, ignore_key = self._get_NTLMv1_response(self._password, self._server_challenge)
    else:
        '\n            [MS-NLMP] v28.0 page 45 - 2016-07-14\n\n            3.1.5.12 Client Received a CHALLENGE_MESSAGE from the Server\n            If NTLMv2 authentication is used and the CHALLENGE_MESSAGE\n            TargetInfo field has an MsvAvTimestamp present, the client SHOULD\n            NOT send the LmChallengeResponse and SHOULD send Z(24) instead.\n            '
        response = self._get_LMv2_response(self._user_name, self._password, self._domain_name, self._server_challenge, self._client_challenge)
        if self._server_target_info is not None:
            timestamp = self._server_target_info[AvId.MSV_AV_TIMESTAMP]
            if timestamp is not None:
                response = b'\x00' * 24
    return response