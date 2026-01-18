import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
@contextlib.contextmanager
def distributed_context(rank, opt, port=61337, rank_offset=0, gpu=None, hostname='localhost'):
    """
    A context which wraps initialization of a distributed/multiprocessing run.

    Every process in the distributed run should launch with this. In true
    distributed setting you may wish to use slurm_distributed_context instead.

    :param int rank:
        This process's rank, less rank_offset.
    :param int rank_offset:
        Used as an offset of rank. Used between multiprocessing vs true distributed,
        and a hack around torch.multiprocessing.spawn being only used for the
        non-primary workers.
    :param opt:
        command line options
    :param int port:
        A TCP port to use. This will need to be changed to run multiple
        distributed training setups on the same machine.
    :param int gpu:
        Which GPU to use. Defaults to using rank and local devices, but must be
        manually specified when using many-hosts.
    :param str hostname:
        Hostname of the main server.
    """
    opt = copy.deepcopy(opt)
    rank = rank + rank_offset
    opt['rank'] = rank
    if gpu is None:
        gpu = rank % torch.cuda.device_count()
    opt['gpu'] = gpu
    if 'override' not in opt:
        opt['override'] = {}
    opt['override']['gpu'] = gpu
    if opt.get('verbose') or rank != 0:
        print_prefix = 'rank:{:3d} |'.format(rank)
    else:
        print_prefix = None
    suppress_output = not opt.get('verbose') and rank != 0
    with override_print(suppress_output, print_prefix):
        if opt['gpu'] != -1:
            torch.cuda.set_device(opt['gpu'])
        dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(hostname, port), world_size=opt['distributed_world_size'], rank=rank)
        logging.info('Distributed group initialized')
        torch.cuda.init()
        torch.manual_seed(42)
        sync_object(None)
        yield opt