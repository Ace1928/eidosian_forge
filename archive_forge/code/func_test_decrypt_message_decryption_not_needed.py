import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_decrypt_message_decryption_not_needed():
    test_session = SessionTest()
    test_response = ResponseTest('application/soap+xml', 'unencrypted message')
    encryption = Encryption(test_session, 'ntlm')
    actual = encryption.parse_encrypted_response(test_response)
    assert actual == 'unencrypted message'