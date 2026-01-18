import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def provider_flavor_capability_list(self, provider):
    """Show the flavor capabilities of the specified provider.

        :param string provider:
            The name of the provider to show
        :return:
            A ``dict`` containing the capabilities of the provider
        """
    url = const.BASE_PROVIDER_FLAVOR_CAPABILITY_URL.format(provider=provider)
    resources = const.PROVIDER_FLAVOR_CAPABILITY_RESOURCES
    response = self._list(url, get_all=True, resources=resources)
    return response