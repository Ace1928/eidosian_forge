from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def start_rabit_tracker(args):
    """Standalone function to start rabit tracker.

    Parameters
    ----------
    args: arguments to start the rabit tracker.
    """
    envs = {'DMLC_NUM_WORKER': args.num_workers, 'DMLC_NUM_SERVER': args.num_servers}
    rabit = RabitTracker(hostIP=get_host_ip(args.host_ip), nslave=args.num_workers)
    envs.update(rabit.slave_envs())
    rabit.start(args.num_workers)
    sys.stdout.write('DMLC_TRACKER_ENV_START\n')
    for k, v in envs.items():
        sys.stdout.write('%s=%s\n' % (k, str(v)))
    sys.stdout.write('DMLC_TRACKER_ENV_END\n')
    sys.stdout.flush()
    rabit.join()