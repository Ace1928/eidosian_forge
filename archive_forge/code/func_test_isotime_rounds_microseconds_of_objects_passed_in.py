import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_isotime_rounds_microseconds_of_objects_passed_in(self):
    time = datetime.datetime.utcnow().replace(microsecond=500000)
    string_time = common_utils.isotime(at=time, subsecond=True)
    expected_string_ending = str(time.second) + '.000000Z'
    self.assertTrue(string_time.endswith(expected_string_ending))