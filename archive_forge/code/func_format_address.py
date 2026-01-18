import select
import socket
from .parser import Parser
from .ports import BaseIOPort, MultiPort
def format_address(host, portno):
    return f'{host}{portno:d}'