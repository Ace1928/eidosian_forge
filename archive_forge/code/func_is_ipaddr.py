import enum
import logging
import logging.handlers
import asyncio
import socket
import queue
import argparse
import yaml
from . import defaults
def is_ipaddr(addr):
    try:
        socket.getaddrinfo(addr, None, flags=socket.AI_NUMERICHOST)
        return True
    except socket.gaierror:
        return False