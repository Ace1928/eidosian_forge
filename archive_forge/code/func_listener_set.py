import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def listener_set(self, listener_id, **kwargs):
    """Update a listener's settings

        :param string listener_id:
            ID of the listener to update
        :param kwargs:
            A dict of arguments to update a listener
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_LISTENER_URL.format(uuid=listener_id)
    response = self._create(url, method='PUT', **kwargs)
    return response