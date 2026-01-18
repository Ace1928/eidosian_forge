from __future__ import annotations
from .threads import (
from ..._compat import cached_property
from .assistants import (
from ..._resource import SyncAPIResource, AsyncAPIResource
from .threads.threads import Threads, AsyncThreads
from .assistants.assistants import Assistants, AsyncAssistants
@cached_property
def assistants(self) -> AsyncAssistantsWithStreamingResponse:
    return AsyncAssistantsWithStreamingResponse(self._beta.assistants)