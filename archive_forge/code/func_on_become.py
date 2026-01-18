from __future__ import (absolute_import, division, print_function)
import re
from abc import ABC, abstractmethod
from ansible.errors import AnsibleConnectionFailure
def on_become(self, passwd=None):
    """Called when privilege escalation is requested

        :kwarg passwd: String containing the password

        This method is called when the privilege is requested to be elevated
        in the play context by setting become to True.  It is the responsibility
        of the terminal plugin to actually do the privilege escalation such
        as entering `enable` mode for instance
        """
    pass