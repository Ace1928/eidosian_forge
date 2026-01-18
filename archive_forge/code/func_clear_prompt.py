import torch
from openchat.base import HuggingfaceAgent
def clear_prompt(self, histories, user_id):
    histories[user_id]['prefix'] = [pf for pf in histories[user_id]['prefix']]