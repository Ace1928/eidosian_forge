import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
@contextlib.contextmanager
def slurm_distributed_context(opt):
    """
    Initialize a distributed context, using the SLURM environment.

    Does some work to read the environment to find a list of participating nodes
    and the main node.

    :param opt:
        Command line options.
    """
    node_list = os.environ.get('SLURM_JOB_NODELIST')
    if node_list is None:
        raise RuntimeError('Does not appear to be in a SLURM environment. You should not call this script directly; see launch_distributed.py')
    try:
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list])
        main_host = hostnames.split()[0].decode('utf-8')
        distributed_rank = int(os.environ['SLURM_PROCID'])
        if opt.get('model_parallel'):
            device_id = -1
        else:
            device_id = int(os.environ['SLURM_LOCALID'])
        port = opt['port']
        logging.info(f'Initializing host {socket.gethostname()} as rank {distributed_rank}, main is {main_host}')
        with distributed_context(distributed_rank, opt, port, 0, device_id, main_host) as opt:
            yield opt
    except subprocess.CalledProcessError as e:
        raise e
    except FileNotFoundError:
        raise RuntimeError('SLURM does not appear to be installed.')