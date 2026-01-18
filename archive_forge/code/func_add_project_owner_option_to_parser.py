from openstack import exceptions
from openstack.identity.v3 import project
from osc_lib.i18n import _
def add_project_owner_option_to_parser(parser):
    """Register project and project domain options.

    :param parser: argparse.Argument parser object.
    """
    parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
    parser.add_argument('--project-domain', metavar='<project-domain>', help=_('Domain the project belongs to (name or ID). This can be used in case collisions between project names exist.'))