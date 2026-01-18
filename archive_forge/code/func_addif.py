import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def addif(self, ctn):
    name = ctn.next_if_name()
    self.ctns.append(ctn)
    ip_address = None
    if self.docker_nw:
        ipv4 = None
        ipv6 = None
        ip_address = self.next_ip_address()
        ip_address_ip = ip_address.split('/')[0]
        version = 4
        if netaddr.IPNetwork(ip_address).version == 6:
            version = 6
        opt_ip = '--ip %s' % ip_address_ip
        if version == 4:
            ipv4 = ip_address
        else:
            opt_ip = '--ip6 %s' % ip_address_ip
            ipv6 = ip_address
        cmd = 'docker network connect %s %s %s' % (opt_ip, self.name, ctn.docker_name())
        self.execute(cmd, sudo=True)
        ctn.set_addr_info(bridge=self.name, ipv4=ipv4, ipv6=ipv6, ifname=name)
    elif self.with_ip:
        ip_address = self.next_ip_address()
        version = 4
        if netaddr.IPNetwork(ip_address).version == 6:
            version = 6
        ctn.pipework(self, ip_address, name, version=version)
    else:
        ctn.pipework(self, '0/0', name)
    return ip_address