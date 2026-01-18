from pprint import pformat
from six import iteritems
import re
@last_update_time.setter
def last_update_time(self, last_update_time):
    """
        Sets the last_update_time of this V1beta2DeploymentCondition.
        The last time this condition was updated.

        :param last_update_time: The last_update_time of this
        V1beta2DeploymentCondition.
        :type: datetime
        """
    self._last_update_time = last_update_time