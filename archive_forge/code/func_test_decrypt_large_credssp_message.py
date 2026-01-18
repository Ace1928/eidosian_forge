import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_decrypt_large_credssp_message():
    test_session = SessionTest()
    test_unencrypted_message = b'unencrypted message ' * 2048
    test_encrypted_message_chunks = [test_unencrypted_message[i:i + 16384] for i in range(0, len(test_unencrypted_message), 16384)]
    test_encrypted_message1 = base64.b64encode(test_encrypted_message_chunks[0])
    test_encrypted_message2 = base64.b64encode(test_encrypted_message_chunks[1])
    test_encrypted_message3 = base64.b64encode(test_encrypted_message_chunks[2])
    test_message = b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-CredSSP-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=16384\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + struct.pack('<i', 5443) + test_encrypted_message1 + b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-CredSSP-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=16384\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + struct.pack('<i', 5443) + test_encrypted_message2 + b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-CredSSP-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=8192\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + struct.pack('<i', 2711) + test_encrypted_message3 + b'--Encrypted Boundary--\r\n'
    test_response = ResponseTest('protocol="application/HTTP-CredSSP-session-encrypted"', test_message)
    encryption = Encryption(test_session, 'credssp')
    actual = encryption.parse_encrypted_response(test_response)
    assert actual == test_unencrypted_message