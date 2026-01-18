import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def health_monitor_set(self, health_monitor_id, **kwargs):
    """Update a health monitor's settings

        :param health_monitor_id:
            ID of the health monitor to update
        :param kwargs:
            A dict of arguments to update a health monitor
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_HEALTH_MONITOR_URL.format(uuid=health_monitor_id)
    response = self._create(url, method='PUT', **kwargs)
    return response