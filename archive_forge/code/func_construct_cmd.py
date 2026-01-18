import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
@staticmethod
def construct_cmd(*parts):
    return ' '.join((str(x) for x in parts))