import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_python_platform():
    import platform
    return platform.platform()