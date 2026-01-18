import datetime
import errno
import logging
import os
import subprocess
import sys
def init_logging():
    LOG.setLevel(logging.INFO)
    LOG.addHandler(logging.StreamHandler())
    fh = logging.FileHandler('/var/log/heat-provision.log')
    os.chmod(fh.baseFilename, int('600', 8))
    LOG.addHandler(fh)