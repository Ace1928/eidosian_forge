from tests.compat import mock, unittest
def client_mock():
    client = client_tmp()
    client.connect = mock.Mock(name='connect')
    return client