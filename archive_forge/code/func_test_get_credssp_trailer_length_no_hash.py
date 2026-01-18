import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_get_credssp_trailer_length_no_hash():
    test_session = SessionTest()
    encryption = Encryption(test_session, 'credssp')
    expected = 2
    actual = encryption._get_credssp_trailer_length(30, 'ECDH-RSA-AES')
    assert actual == expected