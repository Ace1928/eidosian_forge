from collections import deque
from threading import local
def disable_trampoline(self):
    self.trampoline_enabled = False