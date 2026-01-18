import inspect
import re
import six
def get_group_setter(self, group):
    """
        @param group: A valid configuration group
        @type group: str
        @return: The setter method for the configuration group.
        @rtype: method object
        """
    prefix = self.ui_setgroup_method_prefix
    return getattr(self, '%s%s' % (prefix, group))