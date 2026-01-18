import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def pool_set(self, pool_id, **kwargs):
    """Update a pool's settings

        :param pool_id:
            ID of the pool to update
        :param kwargs:
            A dict of arguments to update a pool
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_POOL_URL.format(pool_id=pool_id)
    response = self._create(url, method='PUT', **kwargs)
    return response