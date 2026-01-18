from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def add_reserved_port(port):
    """Add a port that was acquired by means other than the port server."""
    _free_ports.add(port)