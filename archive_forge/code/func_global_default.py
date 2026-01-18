from pprint import pformat
from six import iteritems
import re
@global_default.setter
def global_default(self, global_default):
    """
        Sets the global_default of this V1PriorityClass.
        globalDefault specifies whether this PriorityClass should be considered
        as the default priority for pods that do not have any priority class.
        Only one PriorityClass can be marked as `globalDefault`. However, if
        more than one PriorityClasses exists with their `globalDefault` field
        set to true, the smallest value of such global default PriorityClasses
        will be used as the default priority.

        :param global_default: The global_default of this V1PriorityClass.
        :type: bool
        """
    self._global_default = global_default