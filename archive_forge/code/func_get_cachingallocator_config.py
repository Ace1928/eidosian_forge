import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_cachingallocator_config():
    ca_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    return ca_config