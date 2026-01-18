from __future__ import (absolute_import, division, print_function)
import re
from abc import ABC, abstractmethod
from ansible.errors import AnsibleConnectionFailure
def on_open_shell(self):
    """Called after the SSH session is established

        This method is called right after the invoke_shell() is called from
        the Paramiko SSHClient instance.  It provides an opportunity to setup
        terminal parameters such as disbling paging for instance.
        """
    pass