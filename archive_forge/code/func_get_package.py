import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def get_package(self):
    """Returns the name of the package in the last entry."""
    return self._blocks[0].package