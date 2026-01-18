import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def load_balancer_list(self, **params):
    """List all load balancers

        :param params:
            Parameters to filter on
        :return:
            List of load balancers
        """
    url = const.BASE_LOADBALANCER_URL
    response = self._list(url, get_all=True, resources=const.LOADBALANCER_RESOURCES, **params)
    return response