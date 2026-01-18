import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def next_if_name(self):
    name = 'eth{0}'.format(len(self.eths) + 1)
    self.eths.append(name)
    return name