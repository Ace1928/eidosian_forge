import urllib.request
import sys
from typing import Tuple
def get_hip_ver_num(hip_version):
    hip_version = hip_version.split('.')
    return int(hip_version[0]) * 100 + int(hip_version[1])