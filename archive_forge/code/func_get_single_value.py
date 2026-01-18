import configobj
from . import bedding, cmdline, errors, globbing, osutils
def get_single_value(self, path, preference_name):
    """Get a single preference for a single file.

        :returns: The string preference value, or None.
        """
    for key, value in self.get_selected_items(path, [preference_name]):
        return value
    return None