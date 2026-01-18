import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def quota_reset(self, project_id):
    """Reset a quota

        :param string project_id:
            The ID of the project to reset quotas
        :return:
            ``None``
        """
    url = const.BASE_SINGLE_QUOTA_URL.format(uuid=project_id)
    response = self._delete(url)
    return response