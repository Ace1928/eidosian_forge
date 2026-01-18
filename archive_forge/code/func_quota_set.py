import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def quota_set(self, project_id, **params):
    """Update a quota's settings

        :param string project_id:
            The ID of the project to update
        :param params:
            A ``dict`` of arguments to update project quota
        :return:
            A ``dict`` representing the updated quota
        """
    url = const.BASE_SINGLE_QUOTA_URL.format(uuid=project_id)
    response = self._create(url, method='PUT', **params)
    return response