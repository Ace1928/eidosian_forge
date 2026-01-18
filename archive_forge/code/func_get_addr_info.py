import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_addr_info(self, bridge, ipv=4):
    addrinfo = {}
    if ipv == 4:
        ip_addrs = self.ip_addrs
    elif ipv == 6:
        ip_addrs = self.ip6_addrs
    else:
        return None
    for addr in ip_addrs:
        if addr[2] == bridge:
            addrinfo[addr[1]] = addr[0]
    return addrinfo