from parlai.core.worlds import create_task
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
import random
def get_personas(opt, shared=None):
    if shared and 'personas_list' in shared:
        return shared['personas_list']
    return _load_personas(opt=opt)