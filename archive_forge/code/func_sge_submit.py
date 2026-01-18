from __future__ import absolute_import
import os
import subprocess
from . import tracker
def sge_submit(nworker, nserver, pass_envs):
    """Internal submission function."""
    env_arg = ','.join(['%s="%s"' % (k, str(v)) for k, v in pass_envs.items()])
    cmd = 'qsub -cwd -t 1-%d -S /bin/bash' % (nworker + nserver)
    if args.queue != 'default':
        cmd += '-q %s' % args.queue
    cmd += ' -N %s ' % args.jobname
    cmd += ' -e %s -o %s' % (args.logdir, args.logdir)
    cmd += ' -pe orte %d' % args.vcores
    cmd += ' -v %s,PATH=${PATH}:.' % env_arg
    cmd += ' %s %s' % (runscript, ' '.join(args.command))
    print(cmd)
    subprocess.check_call(cmd, shell=True)
    print('Waiting for the jobs to get up...')