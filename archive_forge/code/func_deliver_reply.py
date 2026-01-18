from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
def deliver_reply(self, raw):
    self._reply = self.REPLY_CLS(raw, huge_tree=self._huge_tree)
    self._reply.set_parsing_error_transform(self._device_handler.reply_parsing_error_transform(self.REPLY_CLS))
    self._event.set()