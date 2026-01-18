from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def add_project_domain_option_to_parser(parser, enhance_help=lambda _h: _h):
    parser.add_argument('--project-domain', metavar='<project-domain>', help=enhance_help(_('Domain the project belongs to (name or ID). This can be used in case collisions between project names exist.')))