import getpass
import hashlib
import sys
from keystoneauth1 import exceptions as ksa_exceptions
from oslo_utils import timeutils
from keystoneclient import exceptions as ksc_exceptions
def prompt_user_password():
    """Prompt user for a password.

    Prompt for a password if stdin is a tty.
    """
    password = None
    if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
        try:
            password = getpass.getpass('Password: ')
        except EOFError:
            pass
    return password