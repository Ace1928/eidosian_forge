import logging
import sys
import threading
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
def forward_stream(fr, to):
    while True:
        line = fr.readline()
        if not line:
            break
        to.write(line)