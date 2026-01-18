import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import
def getJobAndPassword(self):
    """
        If kestrel_options is set to job/password, then return
        the job and password values
        """
    jobNumber = 0
    password = ''
    options = os.getenv('kestrel_options')
    if options is not None:
        m = re.search('job\\s*=\\s*(\\d+)', options, re.IGNORECASE)
        if m:
            jobNumber = int(m.groups()[0])
        m = re.search('password\\s*=\\s*(\\S+)', options, re.IGNORECASE)
        if m:
            password = m.groups()[0]
    return (jobNumber, password)