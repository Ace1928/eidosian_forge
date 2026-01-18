import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def args2body_common(body, parsed_args):
    neutronV20.update_dict(parsed_args, body, ['name', 'description'])