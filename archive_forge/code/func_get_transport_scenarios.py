import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def get_transport_scenarios():
    result = []
    basis = per_transport.transport_test_permutations()
    usable_classes = set()
    if features.paramiko.available():
        from ....transport import sftp
        usable_classes.add(sftp.SFTPTransport)
    from ....transport import local
    usable_classes.add(local.LocalTransport)
    for name, d in basis:
        t_class = d['transport_class']
        if t_class in usable_classes:
            result.append((name, d))
    return result