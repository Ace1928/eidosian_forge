import json
import netaddr
import re
@property
def eth(self):
    """The Ethernet address."""
    return self._eth