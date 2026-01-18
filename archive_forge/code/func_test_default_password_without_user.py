from io import BytesIO
from .... import config, errors, osutils, tests
from .... import transport as _mod_transport
from ... import netrc_credential_store
def test_default_password_without_user(self):
    cs = self._get_netrc_cs()
    password = cs.decode_password(dict(host='other'))
    self.assertIs(None, password)