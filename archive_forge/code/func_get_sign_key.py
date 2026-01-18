import hashlib
import hmac
from ntlm_auth.des import DES
from ntlm_auth.constants import NegotiateFlags
def get_sign_key(exported_session_key, magic_constant):
    """
    3.4.5.2 SIGNKEY

    :param exported_session_key: A 128-bit session key used to derive signing
        and sealing keys
    :param magic_constant: A constant value set in the MS-NLMP documentation
        (constants.SignSealConstants)
    :return sign_key: Key used to sign messages
    """
    sign_key = hashlib.md5(exported_session_key + magic_constant).digest()
    return sign_key