import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def file_prompt(self, s):
    """Expected to return a file name, given"""
    self.request_context = greenlet.getcurrent()
    self.prompt = s
    self.in_prompt = True
    return self.main_context.switch(s)