from pprint import pformat
from six import iteritems
import re
@provider_id.setter
def provider_id(self, provider_id):
    """
        Sets the provider_id of this V1NodeSpec.
        ID of the node assigned by the cloud provider in the format:
        <ProviderName>://<ProviderSpecificNodeID>

        :param provider_id: The provider_id of this V1NodeSpec.
        :type: str
        """
    self._provider_id = provider_id