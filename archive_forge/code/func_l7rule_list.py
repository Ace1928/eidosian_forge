import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def l7rule_list(self, l7policy_id, **kwargs):
    """List all l7rules for a l7policy

        :param kwargs:
            Parameters to filter on
        :return:
            List of l7rules
        """
    url = const.BASE_L7RULE_URL.format(policy_uuid=l7policy_id)
    response = self._list(url, get_all=True, resources=const.L7RULE_RESOURCES, **kwargs)
    return response