from typing import Callable, Optional
def set_on_stop(self, on_stop: Optional[Callable[['TrackedActor'], None]]):
    self._on_stop = on_stop