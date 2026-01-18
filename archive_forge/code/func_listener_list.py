import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def listener_list(self, **kwargs):
    """List all listeners

        :param kwargs:
            Parameters to filter on
        :return:
            List of listeners
        """
    url = const.BASE_LISTENER_URL
    response = self._list(url, get_all=True, resources=const.LISTENER_RESOURCES, **kwargs)
    return response