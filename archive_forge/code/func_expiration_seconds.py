from pprint import pformat
from six import iteritems
import re
@expiration_seconds.setter
def expiration_seconds(self, expiration_seconds):
    """
        Sets the expiration_seconds of this V1ServiceAccountTokenProjection.
        ExpirationSeconds is the requested duration of validity of the service
        account token. As the token approaches expiration, the kubelet volume
        plugin will proactively rotate the service account token. The kubelet
        will start trying to rotate the token if the token is older than 80
        percent of its time to live or if the token is older than 24
        hours.Defaults to 1 hour and must be at least 10 minutes.

        :param expiration_seconds: The expiration_seconds of this
        V1ServiceAccountTokenProjection.
        :type: int
        """
    self._expiration_seconds = expiration_seconds