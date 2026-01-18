import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzoneprofile_delete(self, availabilityzoneprofile_id):
    """Delete a availabilityzone profile

        :param string availabilityzoneprofile_id:
            ID of the availabilityzone profile to delete
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_AVAILABILITYZONEPROFILE_URL.format(uuid=availabilityzoneprofile_id)
    response = self._delete(url)
    return response