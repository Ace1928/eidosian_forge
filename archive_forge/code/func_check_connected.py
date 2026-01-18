from typing import TYPE_CHECKING
from types import SimpleNamespace
def check_connected(self) -> bool:
    return self.worker.ping_server()