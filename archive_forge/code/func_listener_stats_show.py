import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def listener_stats_show(self, listener_id, **kwargs):
    """Shows the current statistics for a listener

        :param string listener_id:
            ID of the listener
        :return:
            A dict of the specified listener's statistics
        """
    url = const.BASE_LISTENER_STATS_URL.format(uuid=listener_id)
    response = self._list(url, **kwargs)
    return response