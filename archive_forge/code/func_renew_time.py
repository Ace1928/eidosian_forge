from pprint import pformat
from six import iteritems
import re
@renew_time.setter
def renew_time(self, renew_time):
    """
        Sets the renew_time of this V1LeaseSpec.
        renewTime is a time when the current holder of a lease has last updated
        the lease.

        :param renew_time: The renew_time of this V1LeaseSpec.
        :type: datetime
        """
    self._renew_time = renew_time