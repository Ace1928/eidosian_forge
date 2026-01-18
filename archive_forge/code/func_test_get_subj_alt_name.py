import os
import mock
import pytest
from ..test_util import TestUtilSSL  # noqa: E402, F401
from ..with_dummyserver.test_https import (  # noqa: E402, F401
from ..with_dummyserver.test_socketlevel import (  # noqa: E402, F401
@mock.patch('urllib3.contrib.pyopenssl.log.warning')
def test_get_subj_alt_name(self, mock_warning):
    """
        If a certificate has two subject alternative names, cryptography raises
        an x509.DuplicateExtension exception.
        """
    path = os.path.join(os.path.dirname(__file__), 'duplicate_san.pem')
    with open(path, 'r') as fp:
        cert = load_certificate(FILETYPE_PEM, fp.read())
    assert get_subj_alt_name(cert) == []
    assert mock_warning.call_count == 1
    assert isinstance(mock_warning.call_args[0][1], x509.DuplicateExtension)