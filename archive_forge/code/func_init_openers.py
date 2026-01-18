import copy
import random
from typing import Any, Dict, List, Optional
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, DialogPartnerWorld, validate
from parlai.core.message import Message
def init_openers(self) -> None:
    """
        Override to load or instantiate opening messages to be used to seed the self
        chat.
        """
    if self.opt.get('seed_messages_from_task'):
        self._openers = load_openers(self.opt)