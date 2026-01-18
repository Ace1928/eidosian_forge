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
def get_agent_for_task(self, task_id: str) -> Optional[ChatServiceAgent]:
    """
        Return ChatServiceAgent for given task id.

        For each "player", a separate agent is created for each task. This
        returns the appropriate MessengerAgent given the task id

        :param task_id:
            task id

        :return:
            ChatServiceAgent object associated with the given task
        """
    if self.has_task(task_id):
        return self.task_id_to_agent[task_id]
    else:
        return None