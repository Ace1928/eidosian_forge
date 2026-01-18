from pprint import pformat
from six import iteritems
import re
@runtime_handler.setter
def runtime_handler(self, runtime_handler):
    """
        Sets the runtime_handler of this V1alpha1RuntimeClassSpec.
        RuntimeHandler specifies the underlying runtime and configuration that
        the CRI implementation will use to handle pods of this class. The
        possible values are specific to the node & CRI configuration.  It is
        assumed that all handlers are available on every node, and handlers of
        the same name are equivalent on every node. For example, a handler
        called "runc" might specify that the runc OCI runtime (using native
        Linux containers) will be used to run the containers in a pod. The
        RuntimeHandler must conform to the DNS Label (RFC 1123) requirements and
        is immutable.

        :param runtime_handler: The runtime_handler of this
        V1alpha1RuntimeClassSpec.
        :type: str
        """
    if runtime_handler is None:
        raise ValueError('Invalid value for `runtime_handler`, must not be `None`')
    self._runtime_handler = runtime_handler