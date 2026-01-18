from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def add_role_domain_option_to_parser(parser):
    parser.add_argument('--role-domain', metavar='<role-domain>', help=_('Domain the role belongs to (name or ID). This must be specified when the name of a domain specific role is used.'))