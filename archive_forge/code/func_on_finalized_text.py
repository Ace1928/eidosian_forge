from queue import Queue
from typing import TYPE_CHECKING, Optional
def on_finalized_text(self, text: str, stream_end: bool=False):
    """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
    self.text_queue.put(text, timeout=self.timeout)
    if stream_end:
        self.text_queue.put(self.stop_signal, timeout=self.timeout)