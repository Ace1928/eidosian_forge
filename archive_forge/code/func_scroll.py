import time
from sys import platform
from typing import (
def scroll(self, direction: str) -> None:
    if direction == 'up':
        self.page.evaluate('(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;')
    elif direction == 'down':
        self.page.evaluate('(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;')