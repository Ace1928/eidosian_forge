from typing import TYPE_CHECKING
from types import SimpleNamespace
@property
def runtime_env(self) -> str:
    return self._fetch_runtime_context().runtime_env