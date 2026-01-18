import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def find_rsrc(resource, name_or_id, cmd_resource=None):
    id_mapping = {'subnet': 'sub1234', 'network': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}
    return id_mapping.get(resource)