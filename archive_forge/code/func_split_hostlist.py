import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def split_hostlist(hostlist):
    """Split hostlist at commas outside of range expressions ('[3-5]')."""
    in_brackets = False
    cur_host = ''
    for c in hostlist:
        if in_brackets:
            assert c != '['
            if c == ']':
                in_brackets = False
        elif c == '[':
            in_brackets = True
        elif c == ',':
            assert cur_host != ''
            yield cur_host
            cur_host = ''
            continue
        cur_host += c
    if cur_host:
        yield cur_host