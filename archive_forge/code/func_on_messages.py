from typing import List, Optional
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
def on_messages(self, messages: List[SequencedMessage]):
    byte_size = 0
    for message in messages:
        byte_size += message.size_bytes
    self._client_tokens += FlowControlRequest(allowed_bytes=-byte_size, allowed_messages=-len(messages))