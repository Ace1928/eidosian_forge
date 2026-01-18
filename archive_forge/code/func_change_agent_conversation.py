import logging
import os
import pickle
import time
from botocore.exceptions import ClientError
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.shared_utils as shared_utils
def change_agent_conversation(self, agent, conversation_id, new_agent_id):
    """
        Handle changing a conversation for an agent, takes a callback for when the
        command is acknowledged.
        """
    agent.id = new_agent_id
    agent.conversation_id = conversation_id
    data = {'text': data_model.COMMAND_CHANGE_CONVERSATION, 'conversation_id': conversation_id, 'agent_id': new_agent_id}
    agent.flush_msg_queue()
    self.mturk_manager.send_command(agent.worker_id, agent.assignment_id, data, ack_func=self._change_agent_to_conv)