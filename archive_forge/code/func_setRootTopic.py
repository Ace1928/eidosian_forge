from __future__ import annotations
import logging
from copy import copy
import zmq
def setRootTopic(self, root_topic: str):
    """Set the root topic for this handler.

        This value is prepended to all messages published by this handler, and it
        defaults to the empty string ''. When you subscribe to this socket, you must
        set your subscription to an empty string, or to at least the first letter of
        the binary representation of this string to ensure you receive any messages
        from this handler.

        If you use the default empty string root topic, messages will begin with
        the binary representation of the log level string (INFO, WARN, etc.).
        Note that ZMQ SUB sockets can have multiple subscriptions.
        """
    if isinstance(root_topic, bytes):
        root_topic = root_topic.decode('utf8')
    self._root_topic = root_topic