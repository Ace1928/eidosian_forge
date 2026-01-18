from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def return_port(port):
    """Return a port that is no longer being used so it can be reused."""
    if port in _random_ports:
        _random_ports.remove(port)
    elif port in _owned_ports:
        _owned_ports.remove(port)
        _free_ports.add(port)
    elif port in _free_ports:
        logging.info('Returning a port that was already returned: %s', port)
    else:
        logging.info("Returning a port that wasn't given by portpicker: %s", port)