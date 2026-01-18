import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def member_set(self, pool_id, member_id, **kwargs):
    """Updating a member settings

        :param pool_id:
            ID of the pool
        :param member_id:
            ID of the member to be updated
        :param kwargs:
            A dict of the values of member to be updated
        :return:
            Response code from the API
        """
    url = const.BASE_SINGLE_MEMBER_URL.format(pool_id=pool_id, member_id=member_id)
    response = self._create(url, method='PUT', **kwargs)
    return response