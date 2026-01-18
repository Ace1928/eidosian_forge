import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
def input_password(self, prompt):
    """Retrieve password from input."""
    return self.pass_from_pipe() or getpass.getpass(prompt)