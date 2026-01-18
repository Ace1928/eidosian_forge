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
def reap_process_and_children(process, timeout=5):
    """Tries hard to terminate and ultimately kill all the children of this process."""

    def on_process_wait_successful(proc):
        returncode = proc.returncode if hasattr(proc, 'returncode') else None
        logger.info('Process {} terminated with exit code {}'.format(proc, returncode))

    def get_process_info(proc):
        return '{}:{}:{} i {}, owner {}'.format(proc.pid, proc.name(), proc.exe(), proc.status(), proc.ppid())
    procs = process.children(recursive=True)[::-1] + [process]
    try:
        logger.info('About to reap process tree of {}, '.format(get_process_info(process)) + 'printing process tree status in termination order:')
        for p in procs:
            logger.info('\t-{}'.format(get_process_info(p)))
    except psutil.ZombieProcess:
        logger.info('Zombie process found in process tree.')
    for p in procs:
        try:
            logger.info('Trying to SIGTERM {}'.format(get_process_info(p)))
            p.terminate()
            try:
                p.wait(timeout=timeout)
                on_process_wait_successful(p)
            except psutil.TimeoutExpired:
                logger.info('Process {} survived SIGTERM; trying SIGKILL on {}'.format(p.pid, get_process_info(p)))
                p.kill()
                try:
                    p.wait(timeout=timeout)
                    on_process_wait_successful(p)
                except psutil.TimeoutExpired:
                    logger.info('Process {} survived SIGKILL; giving up (final status) {}, (owner) {}'.format(p.pid, p.status(), p.ppid()))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            logger.info('Process {} does not exist or is zombie.'.format(p))