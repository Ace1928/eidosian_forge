import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def amphora_list(self, **kwargs):
    """List all amphorae

        :param kwargs:
            Parameters to filter on
        :return:
            A ``dict`` containing a list of amphorae
        """
    url = const.BASE_AMPHORA_URL
    response = self._list(path=url, get_all=True, resources=const.AMPHORA_RESOURCES, **kwargs)
    return response