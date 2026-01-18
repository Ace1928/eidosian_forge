from pprint import pformat
from six import iteritems
import re
@desired_healthy.setter
def desired_healthy(self, desired_healthy):
    """
        Sets the desired_healthy of this V1beta1PodDisruptionBudgetStatus.
        minimum desired number of healthy pods

        :param desired_healthy: The desired_healthy of this
        V1beta1PodDisruptionBudgetStatus.
        :type: int
        """
    if desired_healthy is None:
        raise ValueError('Invalid value for `desired_healthy`, must not be `None`')
    self._desired_healthy = desired_healthy