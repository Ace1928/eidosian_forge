from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def add_known_arguments(self, parser):
    _add_common_args(parser, is_create=False)
    utils.add_boolean_argument(parser, '--admin-state-up', help=_('Specify the administrative state of the policy (True meaning "Up").'))