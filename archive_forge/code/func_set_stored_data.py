import time
from abc import ABC, abstractmethod
from queue import Queue
from parlai.core.agents import Agent
def set_stored_data(self):
    """
        Gets agent state data from manager.
        """
    agent_state = self.manager.get_agent_state(self.id)
    if agent_state is not None and hasattr(agent_state, 'stored_data'):
        self.stored_data = agent_state.stored_data