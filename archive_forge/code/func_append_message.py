import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def append_message(self, message):
    """
        Appends a message to the list of messages, ensures that it is not a duplicate
        message.
        """
    if message['message_id'] in self.message_ids:
        return
    self.message_ids.append(message['message_id'])
    self.messages.append(message)