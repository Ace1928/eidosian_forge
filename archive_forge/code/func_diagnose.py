import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
def diagnose(self):
    config_root = core._config_path()
    if config_root.exists():
        print('config path:', config_root)
    else:
        print('config path:', config_root, '(absent)')
    print('data root:', platform_.data_root())