from abc import ABC, abstractmethod
from dataclasses import dataclass
from openchat.base import BaseAgent, DecoderLM
def make_model_input(self, user_id, user_input, agent):
    prefix = self.histories[user_id]['prefix']
    if len(prefix) > 0:
        prefix = agent.suffix.join(prefix) + agent.suffix
    else:
        prefix = ''
    if isinstance(agent, DecoderLM):
        user_input += agent.suffix
    current_tokens = agent.tokenizer(prefix + user_input)['input_ids']
    histories_for_current_turn = []
    num_history_tokens = len(current_tokens)
    for u, m in zip(reversed(self.histories[user_id]['user_message']), reversed(self.histories[user_id]['bot_message'])):
        history = u + agent.suffix + m + agent.suffix
        tokens = agent.tokenizer(history)['input_ids']
        num_history_tokens += len(tokens)
        if num_history_tokens < agent.maxlen:
            histories_for_current_turn.append(history)
        else:
            break
    histories_for_current_turn = list(reversed(histories_for_current_turn))
    return prefix + ''.join(histories_for_current_turn) + user_input