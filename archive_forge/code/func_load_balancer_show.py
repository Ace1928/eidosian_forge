import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def load_balancer_show(self, lb_id):
    """Show a load balancer

        :param string lb_id:
            ID of the load balancer to show
        :return:
            A dict of the specified load balancer's settings
        """
    response = self._find(path=const.BASE_LOADBALANCER_URL, value=lb_id)
    return response