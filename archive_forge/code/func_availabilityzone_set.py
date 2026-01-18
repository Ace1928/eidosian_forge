import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzone_set(self, availabilityzone_name, **kwargs):
    """Update a availabilityzone's settings

        :param string availabilityzone_name:
            Name of the availabilityzone to update
        :param kwargs:
            A dict of arguments to update a availabilityzone
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_AVAILABILITYZONE_URL.format(name=availabilityzone_name)
    response = self._create(url, method='PUT', **kwargs)
    return response