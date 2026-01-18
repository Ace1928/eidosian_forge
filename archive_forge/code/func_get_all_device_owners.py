from __future__ import (absolute_import, division, print_function)
import collections
import errno
import glob
import json
import os
import re
import sys
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_file_content, get_file_lines, get_mount_size
from ansible.module_utils.six import iteritems
from ansible.module_utils.facts import timeout
def get_all_device_owners(self):
    try:
        retval = collections.defaultdict(set)
        for path in glob.glob('/sys/block/*/slaves/*'):
            elements = path.split('/')
            device = elements[3]
            target = elements[5]
            retval[target].add(device)
        return dict(((k, list(sorted(v))) for k, v in iteritems(retval)))
    except OSError:
        return {}