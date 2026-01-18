from pprint import pformat
from six import iteritems
import re
@denied.setter
def denied(self, denied):
    """
        Sets the denied of this V1SubjectAccessReviewStatus.
        Denied is optional. True if the action would be denied, otherwise false.
        If both allowed is false and denied is false, then the authorizer has no
        opinion on whether to authorize the action. Denied may not be true if
        Allowed is true.

        :param denied: The denied of this V1SubjectAccessReviewStatus.
        :type: bool
        """
    self._denied = denied