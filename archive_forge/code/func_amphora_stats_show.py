import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def amphora_stats_show(self, amphora_id, **kwargs):
    """Show the current statistics for an amphora

        :param string amphora_id:
            ID of the amphora to show
        :return:
            A ``list`` of ``dict`` of the specified amphora's statistics
        """
    url = const.BASE_AMPHORA_STATS_URL.format(uuid=amphora_id)
    response = self._list(path=url, **kwargs)
    return response