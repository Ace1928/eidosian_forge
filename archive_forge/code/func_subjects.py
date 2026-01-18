from pprint import pformat
from six import iteritems
import re
@subjects.setter
def subjects(self, subjects):
    """
        Sets the subjects of this V1ClusterRoleBinding.
        Subjects holds references to the objects the role applies to.

        :param subjects: The subjects of this V1ClusterRoleBinding.
        :type: list[V1Subject]
        """
    self._subjects = subjects