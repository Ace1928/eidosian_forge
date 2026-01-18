from __future__ import annotations
import re
import socket
import struct
import sys
import fastparquet as fp
import numpy as np
import pandas as pd
def get_ip_protocol(s):
    if 'tcp' in s:
        return 'tcp'
    if 'UDP' in s:
        return 'udp'
    if 'EIGRP' in s:
        return 'eigrp'
    if 'ICMP' in s:
        return 'icmp'
    return None