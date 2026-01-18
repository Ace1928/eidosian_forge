import pytest
from urllib3._collections import HTTPHeaderDict
from urllib3._collections import RecentlyUsedContainer as Container
from urllib3.exceptions import InvalidHeader
from urllib3.packages import six
@pytest.mark.skipif(not six.PY2, reason='python3 has a different internal header implementation')
def test_from_httplib_py2(self):
    msg = '\nServer: nginx\nContent-Type: text/html; charset=windows-1251\nConnection: keep-alive\nX-Some-Multiline: asdf\n asdf\t\n\t asdf\nSet-Cookie: bb_lastvisit=1348253375; expires=Sat, 21-Sep-2013 18:49:35 GMT; path=/\nSet-Cookie: bb_lastactivity=0; expires=Sat, 21-Sep-2013 18:49:35 GMT; path=/\nwww-authenticate: asdf\nwww-authenticate: bla\n\n'
    buffer = six.moves.StringIO(msg.lstrip().replace('\n', '\r\n'))
    msg = six.moves.http_client.HTTPMessage(buffer)
    d = HTTPHeaderDict.from_httplib(msg)
    assert d['server'] == 'nginx'
    cookies = d.getlist('set-cookie')
    assert len(cookies) == 2
    assert cookies[0].startswith('bb_lastvisit')
    assert cookies[1].startswith('bb_lastactivity')
    assert d['x-some-multiline'] == 'asdf asdf asdf'
    assert d['www-authenticate'] == 'asdf, bla'
    assert d.getlist('www-authenticate') == ['asdf', 'bla']
    with_invalid_multiline = '\tthis-is-not-a-header: but it has a pretend value\nAuthorization: Bearer 123\n\n'
    buffer = six.moves.StringIO(with_invalid_multiline.replace('\n', '\r\n'))
    msg = six.moves.http_client.HTTPMessage(buffer)
    with pytest.raises(InvalidHeader):
        HTTPHeaderDict.from_httplib(msg)