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
def get_device_links(self, link_dir):
    if not os.path.exists(link_dir):
        return {}
    try:
        retval = collections.defaultdict(set)
        for entry in os.listdir(link_dir):
            try:
                target = os.path.basename(os.readlink(os.path.join(link_dir, entry)))
                retval[target].add(entry)
            except OSError:
                continue
        return dict(((k, list(sorted(v))) for k, v in iteritems(retval)))
    except OSError:
        return {}