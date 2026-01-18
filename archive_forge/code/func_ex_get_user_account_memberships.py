import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def ex_get_user_account_memberships(self):

    def _ex_get_user_account_memberships(params):
        url = '{}/memberships'.format(API_BASE)
        response = self.connection.request(url, params=params)
        return (response, response.object['result'])
    return self._paginate(_ex_get_user_account_memberships, self.MEMBERSHIPS_PAGE_SIZE)