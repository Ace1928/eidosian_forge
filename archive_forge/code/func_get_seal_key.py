import hashlib
import hmac
from ntlm_auth.des import DES
from ntlm_auth.constants import NegotiateFlags
def get_seal_key(negotiate_flags, exported_session_key, magic_constant):
    """
    3.4.5.3. SEALKEY
    Main method to use to calculate the seal_key used to seal (encrypt)
    messages. This will determine the correct method below to use based on the
    compatibility flags set and should be called instead of the others

    :param exported_session_key: A 128-bit session key used to derive signing
        and sealing keys
    :param negotiate_flags: The negotiate_flags structure sent by the server
    :param magic_constant: A constant value set in the MS-NLMP documentation
        (constants.SignSealConstants)
    :return seal_key: Key used to seal messages
    """
    if negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY:
        seal_key = _get_seal_key_ntlm2(negotiate_flags, exported_session_key, magic_constant)
    elif negotiate_flags & NegotiateFlags.NTLMSSP_NEGOTIATE_LM_KEY:
        seal_key = _get_seal_key_ntlm1(negotiate_flags, exported_session_key)
    else:
        seal_key = exported_session_key
    return seal_key