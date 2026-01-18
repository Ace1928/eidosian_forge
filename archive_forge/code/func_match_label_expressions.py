from pprint import pformat
from six import iteritems
import re
@match_label_expressions.setter
def match_label_expressions(self, match_label_expressions):
    """
        Sets the match_label_expressions of this V1TopologySelectorTerm.
        A list of topology selector requirements by labels.

        :param match_label_expressions: The match_label_expressions of this
        V1TopologySelectorTerm.
        :type: list[V1TopologySelectorLabelRequirement]
        """
    self._match_label_expressions = match_label_expressions