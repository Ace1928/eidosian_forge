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
def get_process_info(proc):
    return '{}:{}:{} i {}, owner {}'.format(proc.pid, proc.name(), proc.exe(), proc.status(), proc.ppid())