import torch
import random
import os
import signal
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script
def launch_and_train(opt, port):
    """
    Perform a fork() to many processes.
    """
    spawncontext = torch.multiprocessing.spawn(multiprocess_train, (opt, port, 1), nprocs=opt['distributed_world_size'] - 1, join=False)
    try:
        retval = multiprocess_train(0, opt, port)
        spawncontext.join()
        return retval
    except KeyboardInterrupt:
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGINT)
        raise