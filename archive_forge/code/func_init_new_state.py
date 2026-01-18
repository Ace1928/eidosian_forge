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
def init_new_state(self):
    """
        Prepare for new run.

        Initialize everything in the agent, task, and thread states
        """
    self.agent_pool = {}
    self.messenger_agent_states = {}
    self.task_to_agent_ids = {}
    self.agent_id_to_overworld_future = {}