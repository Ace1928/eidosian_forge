import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def flush_msg_queue(self):
    """
        Clear all messages in the message queue.

        Return flushed messages
        """
    messages = []
    if self.msg_queue is None:
        return []
    while not self.msg_queue.empty():
        messages.append(self.msg_queue.get())
    return messages