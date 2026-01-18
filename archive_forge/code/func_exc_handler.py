import sys
import traceback
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type
import wandb
from wandb.errors import Error
def exc_handler(self, exc_type: Type[BaseException], exc: BaseException, tb: TracebackType) -> None:
    self.exit_code = 1
    self.exception = exc
    if issubclass(exc_type, Error):
        wandb.termerror(str(exc), repeat=False)
    if self.was_ctrl_c():
        self.exit_code = 255
    traceback.print_exception(exc_type, exc, tb)
    if self._orig_excepthook:
        self._orig_excepthook(exc_type, exc, tb)