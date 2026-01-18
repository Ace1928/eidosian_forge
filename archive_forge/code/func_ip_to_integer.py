from __future__ import annotations
import re
import socket
import struct
import sys
import fastparquet as fp
import numpy as np
import pandas as pd
def ip_to_integer(s):
    return struct.unpack('!I', socket.inet_aton(s))[0]