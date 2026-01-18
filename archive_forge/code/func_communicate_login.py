import logging
import time
from abc import abstractmethod
from multiprocessing.process import BaseProcess
from typing import Any, Optional, cast
import wandb
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from wandb.util import json_dumps_safer, json_friendly
from ..lib.mailbox import Mailbox, MailboxHandle
from .interface import InterfaceBase
from .message_future import MessageFuture
from .router import MessageRouter
def communicate_login(self, api_key: Optional[str]=None, timeout: Optional[int]=15) -> pb.LoginResponse:
    login = self._make_login(api_key)
    rec = self._make_request(login=login)
    result = self._communicate(rec, timeout=timeout)
    if result is None:
        raise wandb.Error("Couldn't communicate with backend after %s seconds" % timeout)
    login_response = result.response.login_response
    assert login_response
    return login_response