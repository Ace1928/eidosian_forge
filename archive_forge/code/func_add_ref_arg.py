from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
def add_ref_arg(self, parser):
    parser.add_argument('URI', help='The URI reference for the secret or container.')