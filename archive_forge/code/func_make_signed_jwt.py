import json
import logging
import time
from oauth2client import _helpers
from oauth2client import _pure_python_crypt
def make_signed_jwt(signer, payload, key_id=None):
    """Make a signed JWT.

    See http://self-issued.info/docs/draft-jones-json-web-token.html.

    Args:
        signer: crypt.Signer, Cryptographic signer.
        payload: dict, Dictionary of data to convert to JSON and then sign.
        key_id: string, (Optional) Key ID header.

    Returns:
        string, The JWT for the payload.
    """
    header = {'typ': 'JWT', 'alg': 'RS256'}
    if key_id is not None:
        header['kid'] = key_id
    segments = [_helpers._urlsafe_b64encode(_helpers._json_encode(header)), _helpers._urlsafe_b64encode(_helpers._json_encode(payload))]
    signing_input = b'.'.join(segments)
    signature = signer.sign(signing_input)
    segments.append(_helpers._urlsafe_b64encode(signature))
    logger.debug(str(segments))
    return b'.'.join(segments)