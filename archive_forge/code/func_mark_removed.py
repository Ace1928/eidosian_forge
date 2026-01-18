import logging
import os
from parlai.core.agents import create_agent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.core.socket import ChatServiceMessageSocket
from parlai.chat_service.services.messenger.message_sender import MessageSender
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
def mark_removed(self, agent_id, pageid):
    """
        Mark the agent as removed from the pool.

        Can be overriden to change other metadata linked to agent removal.

        :param agent_id:
            int agent psid
        :param pageid:
            int page id
        """
    pass