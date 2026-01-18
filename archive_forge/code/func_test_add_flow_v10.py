import unittest
from subprocess import Popen, PIPE, STDOUT
import time
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
def test_add_flow_v10(self):
    app = 'os_ken/tests/integrated/test_add_flow_v10.py'
    self._run_os_ken_manager_and_check_output(app)