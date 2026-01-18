from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.utils.misc import display_messages, load_cands
from parlai.utils.strings import colorize
def epoch_done(self):
    return self.finished