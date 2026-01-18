import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_bridges_ovs(self):
    out = self.execute('ovs-vsctl list-br', sudo=True, retry=True)
    return out.splitlines()