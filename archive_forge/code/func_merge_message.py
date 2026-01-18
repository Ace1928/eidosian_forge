import re
import patiencediff
from ... import bugtracker, osutils
def merge_message(self, new_message):
    """Merge new_message with self.message.

        :param new_message: A string message to merge with self.message.
        :return: A string with the merged messages.
        """
    if self.message is None:
        return new_message
    return self.message + new_message