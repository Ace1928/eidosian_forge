import pytest
from mock import Mock, patch
def test_inject_validate_fail_pyopenssl(self):
    """
        Injection should not be supported if pyOpenSSL is too old.
        """
    try:
        return_val = Mock()
        del return_val._x509
        with patch('OpenSSL.crypto.X509', return_value=return_val):
            with pytest.raises(ImportError):
                inject_into_urllib3()
    finally:
        extract_from_urllib3()