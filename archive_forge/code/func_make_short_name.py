import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def make_short_name(long_name):
    if long_name is None:
        return None
    long_name = os.path.abspath(long_name)
    long_dirname = os.path.dirname(long_name)
    tmpdir = os.getenv('TMPDIR', '/tmp')
    for x in range(0, 1000):
        link_name = '%s/ovs-un-py-%d-%d' % (tmpdir, random.randint(0, 10000), x)
        try:
            os.symlink(long_dirname, link_name)
            ovs.fatal_signal.add_file_to_unlink(link_name)
            return os.path.join(link_name, os.path.basename(long_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                break
    raise Exception('Failed to create temporary symlink')