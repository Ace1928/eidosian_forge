from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def number_of_probers(self):
    """Returns the probers value from the monitor string.

        The monitor string for a Require monitor looks like this.

            require 1 from 2 { /Common/tcp }

        This method parses out the first of the numeric values. This values represents
        the "probers" value that can be updated in the module.

        Returns:
             int: The probers value if found. None otherwise.
        """
    if self._values['monitors'] is None:
        return None
    pattern = 'require\\s+\\d+\\s+from\\s+(?P<probers>\\d+)\\s+'
    matches = re.search(pattern, self._values['monitors'])
    if matches is None:
        return None
    return int(matches.group('probers'))