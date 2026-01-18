from pprint import pformat
from six import iteritems
import re
@revision_history_limit.setter
def revision_history_limit(self, revision_history_limit):
    """
        Sets the revision_history_limit of this V1beta2StatefulSetSpec.
        revisionHistoryLimit is the maximum number of revisions that will be
        maintained in the StatefulSet's revision history. The revision history
        consists of all revisions not represented by a currently applied
        StatefulSetSpec version. The default value is 10.

        :param revision_history_limit: The revision_history_limit of this
        V1beta2StatefulSetSpec.
        :type: int
        """
    self._revision_history_limit = revision_history_limit