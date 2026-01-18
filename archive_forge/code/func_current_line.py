import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
@property
def current_line(self):
    self._check_for_expired_message()
    if self.in_prompt:
        return self.prompt + self._current_line
    if self.in_confirm:
        return self.prompt
    if self._message:
        return self._message
    if self.permanent_stack:
        return self.permanent_stack[-1]
    return ''