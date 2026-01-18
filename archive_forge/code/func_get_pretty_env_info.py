import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_pretty_env_info():
    return pretty_str(get_env_info())