import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
def do_set(self):
    password = self.input_password(f"Password for '{self.username}' in '{self.service}': ")
    set_password(self.service, self.username, password)