from pprint import pformat
from six import iteritems
import re
@ca_bundle.setter
def ca_bundle(self, ca_bundle):
    """
        Sets the ca_bundle of this V1APIServiceSpec.
        CABundle is a PEM encoded CA bundle which will be used to validate an
        API server's serving certificate. If unspecified, system trust roots on
        the apiserver are used.

        :param ca_bundle: The ca_bundle of this V1APIServiceSpec.
        :type: str
        """
    if ca_bundle is not None and (not re.search('^(?:[A-Za-z0-9+\\/]{4})*(?:[A-Za-z0-9+\\/]{2}==|[A-Za-z0-9+\\/]{3}=)?$', ca_bundle)):
        raise ValueError('Invalid value for `ca_bundle`, must be a follow pattern or equal to `/^(?:[A-Za-z0-9+\\/]{4})*(?:[A-Za-z0-9+\\/]{2}==|[A-Za-z0-9+\\/]{3}=)?$/`')
    self._ca_bundle = ca_bundle