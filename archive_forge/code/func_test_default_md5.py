from keystoneauth1 import exceptions as ksa_exceptions
import testresources
from testtools import matchers
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils as test_utils
from keystoneclient import utils
def test_default_md5(self):
    """The default hash method is md5."""
    token = self.examples.SIGNED_TOKEN_SCOPED
    token = token.encode('utf-8')
    token_id_default = utils.hash_signed_token(token)
    token_id_md5 = utils.hash_signed_token(token, mode='md5')
    self.assertThat(token_id_default, matchers.Equals(token_id_md5))
    self.assertThat(token_id_default, matchers.HasLength(32))