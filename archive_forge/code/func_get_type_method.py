import inspect
import re
import six
def get_type_method(self, type):
    """
        Returns the type helper method matching the type name.
        """
    return getattr(self, '%s%s' % (self.ui_type_method_prefix, type))