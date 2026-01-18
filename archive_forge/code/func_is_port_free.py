from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def is_port_free(port):
    """Check if specified port is free.

    Args:
      port: integer, port to check

    Returns:
      bool, whether port is free to use for both TCP and UDP.
    """
    return _is_port_free(port)