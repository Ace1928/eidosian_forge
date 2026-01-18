from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def has_action(self, name):
    """
        Check whether the resource has a given action.

        :param name: The name of the action.
        """
    return name in self.actions