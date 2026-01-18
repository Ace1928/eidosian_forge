import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def listener_show(self, listener_id):
    """Show a listener

        :param string listener_id:
            ID of the listener to show
        :return:
            A dict of the specified listener's settings
        """
    response = self._find(path=const.BASE_LISTENER_URL, value=listener_id)
    return response