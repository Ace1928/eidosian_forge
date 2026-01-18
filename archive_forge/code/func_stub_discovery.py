import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def stub_discovery(self, base_url=None, **kwargs):
    kwargs.setdefault('href', self.TEST_URL)
    disc = fixture.DiscoveryList(**kwargs)
    self.stub_url('GET', json=disc, base_url=base_url, status_code=300)
    return disc