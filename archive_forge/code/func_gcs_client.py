from typing import TYPE_CHECKING
from types import SimpleNamespace
@property
def gcs_client(self) -> str:
    return SimpleNamespace(address=self._fetch_runtime_context().gcs_address)