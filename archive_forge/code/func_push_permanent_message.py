import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def push_permanent_message(self, msg):
    self._message = ''
    self.permanent_stack.append(msg)