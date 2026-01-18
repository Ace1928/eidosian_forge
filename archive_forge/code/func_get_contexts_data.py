import json
import random
from parlai.tasks.blended_skill_talk.agents import raw_data_path, safe_personas_path
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
def get_contexts_data(opt, shared=None):
    if shared and 'contexts_data' in shared:
        return shared['contexts_data']
    return _load_personas(opt=opt)