from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def connect_secondary(self):
    return self._validate_connection(self.second_url, self.second_user, self.second_pwd, self.second_ca)