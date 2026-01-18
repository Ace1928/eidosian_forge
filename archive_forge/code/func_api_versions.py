from pprint import pformat
from six import iteritems
import re
@api_versions.setter
def api_versions(self, api_versions):
    """
        Sets the api_versions of this V1beta1RuleWithOperations.
        APIVersions is the API versions the resources belong to. '*' is all
        versions. If '*' is present, the length of the slice must be one.
        Required.

        :param api_versions: The api_versions of this V1beta1RuleWithOperations.
        :type: list[str]
        """
    self._api_versions = api_versions