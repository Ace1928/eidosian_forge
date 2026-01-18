import random
import threading
import time
from .messages import Message
from .parser import Parser
def multi_send(ports, msg):
    """Send message on all ports."""
    for port in ports:
        port.send(msg)