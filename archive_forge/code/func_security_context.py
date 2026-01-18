from pprint import pformat
from six import iteritems
import re
@security_context.setter
def security_context(self, security_context):
    """
        Sets the security_context of this V1Container.
        Security options the pod should run with. More info:
        https://kubernetes.io/docs/concepts/policy/security-context/ More info:
        https://kubernetes.io/docs/tasks/configure-pod-container/security-context/

        :param security_context: The security_context of this V1Container.
        :type: V1SecurityContext
        """
    self._security_context = security_context