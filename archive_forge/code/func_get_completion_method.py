import inspect
import re
import six
def get_completion_method(self, command):
    """
        @return: A user command's completion method or None.
        @rtype: method or None
        @param command: The command to get the completion method for.
        @type command: str
        """
    prefix = self.ui_complete_method_prefix
    try:
        method = getattr(self, '%s%s' % (prefix, command))
    except AttributeError:
        return None
    else:
        return method