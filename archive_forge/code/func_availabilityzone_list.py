import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzone_list(self, **kwargs):
    """List all availabilityzones

        :param kwargs:
            Parameters to filter on
        :return:
            A ``dict`` containing a list of availabilityzone
        """
    url = const.BASE_AVAILABILITYZONE_URL
    response = self._list(path=url, get_all=True, resources=const.AVAILABILITYZONE_RESOURCES, **kwargs)
    return response