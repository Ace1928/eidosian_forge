import sys
import subprocess
import shlex
import os
import argparse
import shutil
import logging
import coloredlogs
def status_print(msg, **kwargs):
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'ignore'), **kwargs)