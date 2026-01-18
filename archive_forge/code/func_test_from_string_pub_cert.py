import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_from_string_pub_cert(self):
    verifier = es256.ES256Verifier.from_string(PUBLIC_CERT_BYTES)
    assert isinstance(verifier, es256.ES256Verifier)
    assert isinstance(verifier._pubkey, ec.EllipticCurvePublicKey)