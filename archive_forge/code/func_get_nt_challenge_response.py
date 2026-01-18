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
def get_nt_challenge_response(self, lm_challenge_response, server_certificate_hash=None, cbt_data=None):
    """
        [MS-NLMP] v28.0 2016-07-14

        3.3.1 - NTLM v1 Authentication
        3.3.2 - NTLM v2 Authentication

        This method returns the NtChallengeResponse key based on the
        ntlm_compatibility chosen and the target_info supplied by the
        CHALLENGE_MESSAGE. It is quite different from what is set in the
        document as it combines the NTLMv1, NTLM2 and NTLMv2 methods into one
        and calls separate methods based on the ntlm_compatibility value
        chosen.

        :param lm_challenge_response: The LmChallengeResponse calculated
            beforehand, used to get the key_exchange_key value
        :param server_certificate_hash: This is deprecated and will be removed
            in a future version, use cbt_data instead
        :param cbt_data: The GssChannelBindingsStruct to bind in the NTLM
            response
        :return response: (NtChallengeResponse) - The NT response to the server
            challenge. Computed by the client
        :return session_base_key: (SessionBaseKey) - A session key calculated
            from the user password challenge
        :return target_info: (AV_PAIR) - The AV_PAIR structure used in the
            nt_challenge calculations
        """
    if self._negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY and self._ntlm_compatibility < 3:
        response, session_base_key = self._get_NTLM2_response(self._password, self._server_challenge, self._client_challenge)
        lm_hash = comphash._lmowfv1(self._password)
        key_exchange_key = compkeys._get_exchange_key_ntlm_v1(self._negotiate_flags, session_base_key, self._server_challenge, lm_challenge_response, lm_hash)
        target_info = None
    elif 0 <= self._ntlm_compatibility < 3:
        response, session_base_key = self._get_NTLMv1_response(self._password, self._server_challenge)
        lm_hash = comphash._lmowfv1(self._password)
        key_exchange_key = compkeys._get_exchange_key_ntlm_v1(self._negotiate_flags, session_base_key, self._server_challenge, lm_challenge_response, lm_hash)
        target_info = None
    else:
        if self._server_target_info is None:
            target_info = ntlm_auth.messages.TargetInfo()
        else:
            target_info = self._server_target_info
        if target_info[AvId.MSV_AV_TIMESTAMP] is None:
            timestamp = get_windows_timestamp()
        else:
            timestamp = target_info[AvId.MSV_AV_TIMESTAMP]
            target_info[AvId.MSV_AV_FLAGS] = struct.pack('<L', AvFlags.MIC_PROVIDED)
        if server_certificate_hash is not None and cbt_data is None:
            certificate_digest = base64.b16decode(server_certificate_hash)
            cbt_data = GssChannelBindingsStruct()
            cbt_data[cbt_data.APPLICATION_DATA] = b'tls-server-end-point:' + certificate_digest
        if cbt_data is not None:
            cbt_bytes = cbt_data.get_data()
            cbt_hash = hashlib.md5(cbt_bytes).digest()
            target_info[AvId.MSV_AV_CHANNEL_BINDINGS] = cbt_hash
        response, session_base_key = self._get_NTLMv2_response(self._user_name, self._password, self._domain_name, self._server_challenge, self._client_challenge, timestamp, target_info)
        key_exchange_key = compkeys._get_exchange_key_ntlm_v2(session_base_key)
    return (response, key_exchange_key, target_info)