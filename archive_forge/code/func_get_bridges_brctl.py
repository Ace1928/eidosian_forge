import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_bridges_brctl(self):
    br_list = []
    bridges = glob.glob('/sys/class/net/*/bridge/bridge_id')
    regex = re.compile('\\/sys\\/class\\/net\\/(.+)\\/bridge\\/bridge_id')
    for bridge in bridges:
        m = regex.match(bridge)
        br_list.append(m.group(1))
    return br_list