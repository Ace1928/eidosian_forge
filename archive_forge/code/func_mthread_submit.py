from __future__ import absolute_import
import sys
import os
import subprocess
import logging
from threading import Thread
from . import tracker
def mthread_submit(nworker, nserver, envs):
    """
        customized submit script, that submit nslave jobs, each must contain args as parameter
        note this can be a lambda function containing additional parameters in input

        Parameters
        ----------
        nworker: number of slave process to start up
        nserver: number of server nodes to start up
        envs: enviroment variables to be added to the starting programs
        """
    procs = {}
    for i in range(nworker + nserver):
        if i < nworker:
            role = 'worker'
        else:
            role = 'server'
        procs[i] = Thread(target=exec_cmd, args=(args.command, args.local_num_attempt, role, i, envs))
        procs[i].setDaemon(True)
        procs[i].start()