from pprint import pformat
from six import iteritems
import re
@stages.setter
def stages(self, stages):
    """
        Sets the stages of this V1alpha1Policy.
        Stages is a list of stages for which events are created.

        :param stages: The stages of this V1alpha1Policy.
        :type: list[str]
        """
    self._stages = stages