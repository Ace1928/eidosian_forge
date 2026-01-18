import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
@staticmethod
def generate_params(argument, params):
    """Generate parameters string.

        :param argument: argument
        :param params: values passed with argument
        """
    parts = []
    for key, value in params.items():
        parts.append('{} {}={}'.format(argument, key, value))
    return ' '.join(parts)