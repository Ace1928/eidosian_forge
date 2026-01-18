from abc import ABC, abstractmethod
from asyncio import Future
import copy
import sys
import logging
import datetime
import threading
import time
import traceback
from typing import Dict, Any, Optional, List, Callable
from parlai.chat_service.core.agents import ChatServiceAgent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.core.world_runner import ChatServiceWorldRunner
from parlai.core.opt import Opt
def remove_agent_from_pool(self, agent: AgentState, world_type: str='default', mark_removed: bool=True):
    """
        Remove agent from the pool.

        :param agent:
            AgentState object
        :param world_type:
            world name
        :param mark_removed:
            whether to mark an agent as removed from the pool
        """
    with self.agent_pool_change_condition:
        self._log_debug('Removing agent {} from pool...'.format(agent.service_id))
        if world_type in self.agent_pool and agent in self.agent_pool[world_type]:
            self.agent_pool[world_type].remove(agent)
            if world_type in agent.time_in_pool:
                del agent.time_in_pool[world_type]
            if mark_removed:
                agent.stored_data['removed_from_pool'] = True
                if self.service_reference_id is not None:
                    self.mark_removed(agent.service_id, self.service_reference_id)