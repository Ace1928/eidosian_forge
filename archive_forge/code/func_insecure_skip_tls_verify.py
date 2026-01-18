from pprint import pformat
from six import iteritems
import re
@insecure_skip_tls_verify.setter
def insecure_skip_tls_verify(self, insecure_skip_tls_verify):
    """
        Sets the insecure_skip_tls_verify of this V1APIServiceSpec.
        InsecureSkipTLSVerify disables TLS certificate verification when
        communicating with this server. This is strongly discouraged.  You
        should use the CABundle instead.

        :param insecure_skip_tls_verify: The insecure_skip_tls_verify of this
        V1APIServiceSpec.
        :type: bool
        """
    self._insecure_skip_tls_verify = insecure_skip_tls_verify