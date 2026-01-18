import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def avx_default():
    if not _os_supports_avx():
        return False
    else:
        cpu_name = ll.get_host_cpu_name()
        return cpu_name not in ('corei7-avx', 'core-avx-i', 'sandybridge', 'ivybridge')