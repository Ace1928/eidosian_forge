import inspect
import logging
import stevedore
def add_legacy_command(self, old_name, new_name):
    """Map an old command name to the new name.

        :param old_name: The old command name.
        :type old_name: str
        :param new_name: The new command name.
        :type new_name: str
        """
    self._legacy[old_name] = new_name