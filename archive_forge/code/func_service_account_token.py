from pprint import pformat
from six import iteritems
import re
@service_account_token.setter
def service_account_token(self, service_account_token):
    """
        Sets the service_account_token of this V1VolumeProjection.
        information about the serviceAccountToken data to project

        :param service_account_token: The service_account_token of this
        V1VolumeProjection.
        :type: V1ServiceAccountTokenProjection
        """
    self._service_account_token = service_account_token