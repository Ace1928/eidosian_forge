from __future__ import absolute_import
import os
import sys
import json
import uuid
import logging
from threading import Thread
from . import tracker
def mesos_submit(nworker, nserver, pass_envs):
    """
        customized submit script
        """
    for i in range(nworker + nserver):
        resources = {}
        pass_envs['DMLC_ROLE'] = 'server' if i < nserver else 'worker'
        if i < nserver:
            pass_envs['DMLC_SERVER_ID'] = i
            resources['cpus'] = args.server_cores
            resources['mem'] = args.server_memory_mb
        else:
            pass_envs['DMLC_WORKER_ID'] = i - nserver
            resources['cpus'] = args.worker_cores
            resources['mem'] = args.worker_memory_mb
        env = {str(k): str(v) for k, v in pass_envs.items()}
        env.update(get_env())
        prog = ' '.join(args.command)
        thread = Thread(target=_run, args=(prog, env, resources))
        thread.setDaemon(True)
        thread.start()
    return mesos_submit