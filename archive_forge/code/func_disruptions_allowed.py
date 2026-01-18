from pprint import pformat
from six import iteritems
import re
@disruptions_allowed.setter
def disruptions_allowed(self, disruptions_allowed):
    """
        Sets the disruptions_allowed of this V1beta1PodDisruptionBudgetStatus.
        Number of pod disruptions that are currently allowed.

        :param disruptions_allowed: The disruptions_allowed of this
        V1beta1PodDisruptionBudgetStatus.
        :type: int
        """
    if disruptions_allowed is None:
        raise ValueError('Invalid value for `disruptions_allowed`, must not be `None`')
    self._disruptions_allowed = disruptions_allowed