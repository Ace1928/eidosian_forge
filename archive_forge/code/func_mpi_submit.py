from __future__ import absolute_import
import sys
import subprocess, logging
from threading import Thread
from . import tracker
def mpi_submit(nworker, nserver, pass_envs):
    """Internal closure for job submission."""

    def run(prog):
        """run the program"""
        subprocess.check_call(prog, shell=True)
    cmd = ''
    if args.host_file is not None:
        cmd = '--hostfile %s ' % args.host_file
    cmd += ' ' + ' '.join(args.command)
    pass_envs['DMLC_JOB_CLUSTER'] = 'mpi'
    if nworker > 0:
        logging.info('Start %d workers by mpirun' % nworker)
        pass_envs['DMLC_ROLE'] = 'worker'
        if sys.platform == 'win32':
            prog = 'mpiexec -n %d %s %s' % (nworker, get_mpi_env(pass_envs), cmd)
        else:
            prog = 'mpirun -n %d %s %s' % (nworker, get_mpi_env(pass_envs), cmd)
        thread = Thread(target=run, args=(prog,))
        thread.setDaemon(True)
        thread.start()
    if nserver > 0:
        logging.info('Start %d servers by mpirun' % nserver)
        pass_envs['DMLC_ROLE'] = 'server'
        if sys.platform == 'win32':
            prog = 'mpiexec -n %d %s %s' % (nserver, get_mpi_env(pass_envs), cmd)
        else:
            prog = 'mpirun -n %d %s %s' % (nserver, get_mpi_env(pass_envs), cmd)
        thread = Thread(target=run, args=(prog,))
        thread.setDaemon(True)
        thread.start()