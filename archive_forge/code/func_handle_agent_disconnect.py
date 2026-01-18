import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def handle_agent_disconnect(self, worker_id, assignment_id, partner_callback):
    """
        Handles a disconnect by the given worker, calls partner_callback on all of the
        conversation partners of that worker.
        """
    agent = self._get_agent(worker_id, assignment_id)
    if agent is not None:
        agent.set_status(AssignState.STATUS_DISCONNECT)
        conversation_id = agent.conversation_id
        if conversation_id in self.conv_to_agent:
            conv_participants = self.conv_to_agent[conversation_id]
            if agent in conv_participants:
                for other_agent in conv_participants:
                    if agent.assignment_id != other_agent.assignment_id:
                        partner_callback(other_agent)
            if len(conv_participants) > 1:
                self.handle_bad_disconnect(worker_id)