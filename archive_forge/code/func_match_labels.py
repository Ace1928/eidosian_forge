from pprint import pformat
from six import iteritems
import re
@match_labels.setter
def match_labels(self, match_labels):
    """
        Sets the match_labels of this V1LabelSelector.
        matchLabels is a map of {key,value} pairs. A single {key,value} in the
        matchLabels map is equivalent to an element of matchExpressions, whose
        key field is "key", the operator is "In", and the values array
        contains only "value". The requirements are ANDed.

        :param match_labels: The match_labels of this V1LabelSelector.
        :type: dict(str, str)
        """
    self._match_labels = match_labels