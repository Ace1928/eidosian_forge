import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzoneprofile_create(self, **kwargs):
    """Create a availabilityzone profile

        :param kwargs:
            Parameters to create a availabilityzone profile with
            (expects json=)
        :return:
            A dict of the created availabilityzone profile's settings
        """
    url = const.BASE_AVAILABILITYZONEPROFILE_URL
    response = self._create(url, **kwargs)
    return response