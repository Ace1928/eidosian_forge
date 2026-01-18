import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def confirm(self, q):
    """Expected to return True or False, given question prompt q"""
    self.request_context = greenlet.getcurrent()
    self.prompt = q
    self.in_confirm = True
    return self.main_context.switch(q)