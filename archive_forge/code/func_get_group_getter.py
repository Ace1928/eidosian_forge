import inspect
import re
import six
def get_group_getter(self, group):
    """
        @param group: A valid configuration group
        @type group: str
        @return: The getter method for the configuration group.
        @rtype: method object
        """
    prefix = self.ui_getgroup_method_prefix
    return getattr(self, '%s%s' % (prefix, group))