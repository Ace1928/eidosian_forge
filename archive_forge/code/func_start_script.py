import logging
import socket
from subprocess import PIPE
from subprocess import Popen
import sys
import time
import traceback
import requests
from saml2test.check import CRITICAL
def start_script(path, *args):
    popen_args = [path]
    popen_args.extend(args)
    return Popen(popen_args, stdout=PIPE, stderr=PIPE)