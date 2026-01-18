from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def add_security_group(self, id, security_group):
    return self._action(id, '/add_security_group', qparams={'name': security_group})