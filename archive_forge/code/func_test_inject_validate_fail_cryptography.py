import pytest
from mock import Mock, patch
def test_inject_validate_fail_cryptography(self):
    """
        Injection should not be supported if cryptography is too old.
        """
    try:
        with patch('cryptography.x509.extensions.Extensions') as mock:
            del mock.get_extension_for_class
            with pytest.raises(ImportError):
                inject_into_urllib3()
    finally:
        extract_from_urllib3()