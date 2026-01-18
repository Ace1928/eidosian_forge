import copy
import random
from typing import Any, Dict, List, Optional
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, DialogPartnerWorld, validate
from parlai.core.message import Message
def load_openers(opt) -> Optional[List[str]]:
    base_task = opt['task'].split(':')[0]
    if base_task == 'self_chat':
        return None
    print('[ loading conversation openers... ]')
    task_opt = copy.deepcopy(opt)
    task_opt['task'] = base_task
    datatype = task_opt['datatype']
    if 'train' in datatype and 'evalmode' not in datatype:
        task_opt['datatype'] = f'{datatype}:evalmode'
    task_opt['interactive_task'] = False
    task_opt['selfchat_task'] = False
    task_agent = RepeatLabelAgent(task_opt)
    task_world = create_task(task_opt, task_agent)
    openers = set()
    is_first_turn = True
    while not task_world.epoch_done():
        task_world.parley()
        msg = task_world.get_acts()[0]
        if is_first_turn and msg.get('text'):
            openers.add(msg['text'])
        is_first_turn = msg.get('episode_done', False)
    print(f'[ loaded {len(openers)} openers ]')
    return list(openers)