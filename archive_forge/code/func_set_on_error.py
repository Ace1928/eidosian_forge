from typing import Callable, Optional
def set_on_error(self, on_error: Optional[Callable[['TrackedActor', Exception], None]]):
    self._on_error = on_error