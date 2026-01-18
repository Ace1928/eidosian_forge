import copy
import random
from typing import Any, Dict, List, Optional
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, DialogPartnerWorld, validate
from parlai.core.message import Message
def make_agent_action(utterance: str, agent: Agent) -> Dict[str, Any]:
    return {'text': utterance, 'episode_done': False, 'id': agent.id}