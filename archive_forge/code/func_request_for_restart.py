from typing import List, Optional
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
def request_for_restart(self) -> Optional[FlowControlRequest]:
    self._pending_tokens = _AggregateRequest()
    return self._client_tokens.to_optional()