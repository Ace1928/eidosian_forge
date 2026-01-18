import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import mtls
@mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
@mock.patch('google.auth.transport.mtls.has_default_client_cert_source', autospec=True)
def test_default_client_cert_source(has_default_client_cert_source, get_client_cert_and_key):
    has_default_client_cert_source.return_value = False
    with pytest.raises(exceptions.MutualTLSChannelError):
        mtls.default_client_cert_source()
    has_default_client_cert_source.return_value = True
    get_client_cert_and_key.return_value = (True, b'cert', b'key')
    callback = mtls.default_client_cert_source()
    assert callback() == (b'cert', b'key')
    get_client_cert_and_key.side_effect = ValueError()
    callback = mtls.default_client_cert_source()
    with pytest.raises(exceptions.MutualTLSChannelError):
        callback()