import copy
import errno
import os
import sys
import ovs.dirs
import ovs.jsonrpc
import ovs.stream
import ovs.unixctl
import ovs.util
import ovs.version
import ovs.vlog
def reply_error(self, body):
    self._reply_impl(False, body)