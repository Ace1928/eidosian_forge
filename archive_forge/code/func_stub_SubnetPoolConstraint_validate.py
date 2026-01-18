import os
import sys
import fixtures
from oslo_config import cfg
from oslo_log import log as logging
import testscenarios
import testtools
from heat.common import context
from heat.common import messaging
from heat.common import policy
from heat.engine.clients.os import barbican
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import keystone_constraints as ks_constr
from heat.engine.clients.os.neutron import neutron_constraints as neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.clients.os import trove
from heat.engine import environment
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def stub_SubnetPoolConstraint_validate(self):
    validate = self.patchobject(neutron.SubnetPoolConstraint, 'validate')
    validate.return_value = True