import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzoneprofile_list(self, **kwargs):
    """List all availabilityzone profiles

        :param kwargs:
            Parameters to filter on
        :return:
            List of availabilityzone profile
        """
    url = const.BASE_AVAILABILITYZONEPROFILE_URL
    resources = const.AVAILABILITYZONEPROFILE_RESOURCES
    response = self._list(url, get_all=True, resources=resources, **kwargs)
    return response