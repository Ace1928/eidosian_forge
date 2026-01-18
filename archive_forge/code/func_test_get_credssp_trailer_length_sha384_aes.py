import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_get_credssp_trailer_length_sha384_aes():
    test_session = SessionTest()
    encryption = Encryption(test_session, 'credssp')
    expected = 50
    actual = encryption._get_credssp_trailer_length(30, 'ECDH-RSA-AES-SHA384')
    assert actual == expected