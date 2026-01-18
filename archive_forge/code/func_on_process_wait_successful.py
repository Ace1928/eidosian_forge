import coloredlogs
import argparse
import psutil
import time
import shutil
import logging
import psutil
import subprocess
import os
import sys
import signal
from daemoniker import daemonize
def on_process_wait_successful(proc):
    returncode = proc.returncode if hasattr(proc, 'returncode') else None
    logger.info('Process {} terminated with exit code {}'.format(proc, returncode))