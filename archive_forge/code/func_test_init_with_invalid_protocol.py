import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_init_with_invalid_protocol():
    with pytest.raises(WinRMError) as excinfo:
        Encryption(None, 'invalid_protocol')
    assert "Encryption for protocol 'invalid_protocol' not supported in pywinrm" in str(excinfo.value)