import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def amphora_show(self, amphora_id):
    """Show an amphora

        :param string amphora_id:
            ID of the amphora to show
        :return:
            A ``dict`` of the specified amphora's attributes
        """
    url = const.BASE_AMPHORA_URL
    response = self._find(path=url, value=amphora_id)
    return response