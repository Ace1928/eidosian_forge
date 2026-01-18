import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
def batch_observe(self, index, batch_actions, index_acting):
    """
        Observe corresponding actions in all subworlds.
        """
    batch_observations = []
    for i, w in enumerate(self.worlds):
        agents = w.get_agents()
        observation = None
        if batch_actions[i] is None:
            batch_actions[i] = [{}] * len(self.worlds)
        if hasattr(w, 'observe'):
            observation = w.observe(agents[index], validate(batch_actions[i]))
        else:
            observation = validate(batch_actions[i])
        if index == index_acting:
            if hasattr(agents[index], 'self_observe'):
                agents[index].self_observe(observation)
        else:
            observation = agents[index].observe(observation)
        if observation is None:
            raise ValueError('Agents should return what they observed.')
        batch_observations.append(observation)
    return batch_observations