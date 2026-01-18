import pytest
from winrm import Session
def test_decode_clixml_invalid_xml():
    s = Session('windows-host.example.com', auth=('john.smith', 'secret'))
    msg = b'#< CLIXML\r\n<in >dasf<?dsfij>'
    with pytest.warns(UserWarning, match='There was a problem converting the Powershell error message'):
        actual = s._clean_error_msg(msg)
    assert actual == msg