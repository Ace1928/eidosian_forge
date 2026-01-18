import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def set_distributions(self, distributions):
    self._blocks[0].distributions = distributions