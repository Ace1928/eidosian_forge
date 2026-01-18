import sys
import threading
from typing import Callable, List, Sequence, TextIO
from absl import logging
def list_message_callback(sink: List[str]) -> SolveMessageCallback:
    """Returns a message callback that logs messages to a list.

    Args:
      sink: The list to append messages to.

    Returns:
      A function matching the expected signature for message callbacks.
    """
    mutex = threading.Lock()

    def callback(messages: Sequence[str]) -> None:
        with mutex:
            for message in messages:
                sink.append(message)
    return callback