from pprint import pformat
from six import iteritems
import re
@match_fields.setter
def match_fields(self, match_fields):
    """
        Sets the match_fields of this V1NodeSelectorTerm.
        A list of node selector requirements by node's fields.

        :param match_fields: The match_fields of this V1NodeSelectorTerm.
        :type: list[V1NodeSelectorRequirement]
        """
    self._match_fields = match_fields