from pprint import pformat
from six import iteritems
import re
@lifecycle.setter
def lifecycle(self, lifecycle):
    """
        Sets the lifecycle of this V1Container.
        Actions that the management system should take in response to container
        lifecycle events. Cannot be updated.

        :param lifecycle: The lifecycle of this V1Container.
        :type: V1Lifecycle
        """
    self._lifecycle = lifecycle