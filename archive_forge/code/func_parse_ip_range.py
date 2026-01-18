from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
def parse_ip_range(self, ip_range):
    if self.ip_regex.match(ip_range) is not None:
        return self.messages.IpRange(ipAddress=ip_range)
    if self.ip_ranges_regex.match(ip_range) is not None:
        return self.messages.IpRange(ipAddressRange=ip_range)
    return self.messages.IpRange(externalAddress=ip_range)