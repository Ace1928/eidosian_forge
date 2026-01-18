import logging
from parlai.mturk.core.agents import MTurkAgent, TIMEOUT_MESSAGE
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.shared_utils as shared_utils
def get_update_packet(self):
    """
        Produce an update packet that represents the state change of this agent.
        """
    send_messages = []
    while len(self.unread_messages) > 0:
        pkt = self.unread_messages.pop(0)
        send_messages.append(pkt.data)
    done_text = None
    if self.state.is_final() and self.get_status() != AssignState.STATUS_DONE:
        done_text = self.state.get_inactive_command_text()[0]
    return {'new_messages': send_messages, 'all_messages': self.state.get_messages(), 'wants_message': self.wants_message, 'disconnected': self.disconnected, 'agent_id': self.id, 'worker_id': self.worker_id, 'conversation_id': self.conversation_id, 'task_done': self.state.is_final(), 'done_text': done_text, 'status': self.state.get_status()}