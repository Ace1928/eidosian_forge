import collections
import copy
from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def mock_no_neutron(self):
    self.patchobject(resource.Resource, 'is_using_neutron', return_value=False)