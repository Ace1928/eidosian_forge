import inspect
import re
import six
def get_command_description(self, command):
    """
        @return: An description string for a user command.
        @rtype: str
        @param command: The command to describe.
        @type command: str
        """
    doc = self.get_command_method(command).__doc__
    if not doc:
        doc = 'No description available.'
    return self.shell.con.dedent(doc)