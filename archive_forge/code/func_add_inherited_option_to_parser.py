from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def add_inherited_option_to_parser(parser):
    parser.add_argument('--inherited', action='store_true', default=False, help=_('Specifies if the role grant is inheritable to the sub projects'))