from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def get_opts(self, fields=None, output_format='json'):
    """Get options for OSC output fields format.

        :param List fields: List of fields to get
        :param String output_format: Select output format
        :return: String of formatted options
        """
    if not fields:
        return ' -f {0}'.format(output_format)
    return ' -f {0} {1}'.format(output_format, ' '.join(['-c ' + it for it in fields]))