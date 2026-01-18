import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def l7rule_set(self, l7rule_id, l7policy_id, **kwargs):
    """Update a l7rule's settings

        :param l7rule_id:
            ID of the l7rule to update
        :param string l7policy_id:
            ID of the l7policy for this l7rule
        :param kwargs:
            A dict of arguments to update a l7rule
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_L7RULE_URL.format(rule_uuid=l7rule_id, policy_uuid=l7policy_id)
    response = self._create(url, method='PUT', **kwargs)
    return response