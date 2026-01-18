import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def set_addr_info(self, bridge, ipv4=None, ipv6=None, ifname='eth0'):
    if ipv4:
        self.ip_addrs.append((ifname, ipv4, bridge))
    if ipv6:
        self.ip6_addrs.append((ifname, ipv6, bridge))