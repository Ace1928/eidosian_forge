import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_missing_parameters(self):
    self.assertRaises(exceptions.OptionError, self.create, domain_id=None)
    self.assertRaises(exceptions.OptionError, self.create, domain_name=None)
    self.assertRaises(exceptions.OptionError, self.create, project_id=None)
    self.assertRaises(exceptions.OptionError, self.create, project_name=None)
    self.assertRaises(exceptions.OptionError, self.create, project_domain_id=None)
    self.assertRaises(exceptions.OptionError, self.create, project_domain_name=None)
    self.assertRaises(exceptions.OptionError, self.create, project_domain_id=uuid.uuid4().hex)
    self.assertRaises(exceptions.OptionError, self.create, project_domain_name=uuid.uuid4().hex)
    self.assertRaises(exceptions.OptionError, self.create, project_name=uuid.uuid4().hex)