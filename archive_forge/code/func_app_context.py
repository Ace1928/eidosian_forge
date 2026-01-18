from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def app_context(self, name=None, global_conf=None):
    return self.get_context(APP, name=name, global_conf=global_conf)