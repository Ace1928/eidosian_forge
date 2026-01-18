import os
import tempfile
from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import huawei
from os_brick.tests.initiator import test_connector
def fake_execute_fail(self, *cmd, **kwargs):
    method = cmd[2]
    self.cmds.append(' '.join(cmd))
    if 'attach' == method:
        HuaweiStorHyperConnectorTestCase.attached = False
        return ('ret_code=330151401', None)
    if 'querydev' == method:
        if HuaweiStorHyperConnectorTestCase.attached:
            return ('ret_code=0\ndev_addr=/dev/vdxxx', None)
        else:
            return ('ret_code=1\ndev_addr=/dev/vdxxx', None)
    if 'detach' == method:
        HuaweiStorHyperConnectorTestCase.attached = True
        return ('ret_code=330155007', None)