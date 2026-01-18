from .symbols import (
from ..schema import extract_record_type
def drain_actions(self):
    while True:
        top = self.stack.pop()
        if isinstance(top, Root):
            self.push_symbol(top)
            break
        elif isinstance(top, Action):
            self.action_function(top)
        elif not isinstance(top, Terminal):
            self.stack.extend(top.production)
        else:
            raise Exception(f'Internal Parser Exception: {top}')