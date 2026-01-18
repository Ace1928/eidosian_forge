from __future__ import annotations
from .speech import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
from .transcriptions import (
@cached_property
def speech(self) -> AsyncSpeechWithStreamingResponse:
    return AsyncSpeechWithStreamingResponse(self._audio.speech)