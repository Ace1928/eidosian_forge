from __future__ import annotations
import os
import typing
from typing import Any, Optional  # noqa: H301
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator import linuxfc
from os_brick import utils
def set_execute(self, execute) -> None:
    super(FibreChannelConnector, self).set_execute(execute)
    self._linuxscsi.set_execute(execute)
    self._linuxfc.set_execute(execute)