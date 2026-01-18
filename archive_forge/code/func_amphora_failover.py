import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def amphora_failover(self, amphora_id):
    """Force failover an amphorae

        :param string amphora_id:
            ID of the amphora to failover
        :return:
            Response Code from the API
        """
    url = const.BASE_AMPHORA_FAILOVER_URL.format(uuid=amphora_id)
    response = self._create(url, method='PUT')
    return response