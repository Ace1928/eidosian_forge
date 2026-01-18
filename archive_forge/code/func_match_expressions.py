from pprint import pformat
from six import iteritems
import re
@match_expressions.setter
def match_expressions(self, match_expressions):
    """
        Sets the match_expressions of this V1LabelSelector.
        matchExpressions is a list of label selector requirements. The
        requirements are ANDed.

        :param match_expressions: The match_expressions of this V1LabelSelector.
        :type: list[V1LabelSelectorRequirement]
        """
    self._match_expressions = match_expressions