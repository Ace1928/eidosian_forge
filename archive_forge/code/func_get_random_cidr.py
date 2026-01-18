import os
import platform
import random
import time
import netaddr
from neutron_lib.utils import helpers
from neutron_lib.utils import net
def get_random_cidr(version=4):
    if version == 4:
        return '10.%d.%d.0/%d' % (random.randint(3, 254), random.randint(3, 254), 24)
    return '2001:db8:%x::/%d' % (random.getrandbits(16), 64)