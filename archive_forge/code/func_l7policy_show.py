import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def l7policy_show(self, l7policy_id):
    """Show a l7policy's settings

        :param string l7policy_id:
            ID of the l7policy to show
        :return:
            Dict of the specified l7policy's settings
        """
    response = self._find(path=const.BASE_L7POLICY_URL, value=l7policy_id)
    return response