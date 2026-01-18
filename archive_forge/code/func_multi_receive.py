import random
import threading
import time
from .messages import Message
from .parser import Parser
def multi_receive(ports, yield_ports=False, block=True):
    """Receive messages from multiple ports.

    Generates messages from ever input port. The ports are polled in
    random order for fairness, and all messages from each port are
    yielded before moving on to the next port.

    If yield_ports=True, (port, message) is yielded instead of just
    the message.

    If block=False only pending messages will be yielded.
    """
    ports = list(ports)
    while True:
        random.shuffle(ports)
        for port in ports:
            if not port.closed:
                for message in port.iter_pending():
                    if yield_ports:
                        yield (port, message)
                    else:
                        yield message
        if block:
            sleep()
        else:
            break