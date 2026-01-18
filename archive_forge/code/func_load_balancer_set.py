import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def load_balancer_set(self, lb_id, **params):
    """Update a load balancer's settings

        :param string lb_id:
            The ID of the load balancer to update
        :param params:
            A dict of arguments to update a loadbalancer
        :return:
            Response Code from API
        """
    url = const.BASE_SINGLE_LB_URL.format(uuid=lb_id)
    response = self._create(url, method='PUT', **params)
    return response