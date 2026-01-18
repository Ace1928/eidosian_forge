import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
def do_test_on_master(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool, remote_em_rref: rpc.RRef, remote_net_rref: rpc.RRef):
    if simulate_uneven_inputs:
        gLogger.info('Running DDP + RPC test with simulating uneven inputs across trainers.')
    trainer_rrefs = []
    for rank in TRAINER_RANKS:
        trainer = self.trainer_name(rank)
        trainer_rrefs.append(rpc.remote(trainer, Trainer, args=(remote_em_rref, remote_net_rref, ddp_mode, rank)))
    if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
        dist.new_group(TRAINER_RANKS)
    training_examples = get_training_examples()
    for _ in range(3):
        futures = []
        num_trainers = len(trainer_rrefs)
        for idx, trainer_rref in enumerate(trainer_rrefs):
            trainer_has_less_inputs = simulate_uneven_inputs and idx < num_trainers // 2
            futures.append(_remote_method_async(Trainer.train_batch, trainer_rref, training_examples[idx], trainer_has_less_inputs, simulate_uneven_inputs))
        for future in futures:
            ddp_grads, non_ddp_grads = future.wait()
            if not simulate_uneven_inputs:
                for grad in ddp_grads:
                    self.assertEqual(grad, torch.zeros_like(grad), msg=f"The grad for any ddp parameter should be zeros, because the training examples' grads cancel each other. Received gradient {grad}")
            for grad in non_ddp_grads:
                self.assertNotEqual(grad, torch.zeros_like(grad), msg="The grad for any non-ddp parameter shouldn't be zeros")
    for idx, trainer_rref in enumerate(trainer_rrefs):
        _remote_method_async(Trainer.destroy_pg, trainer_rref).wait()
    for rank in TRAINER_RANKS:
        trainer = self.trainer_name(rank)
        rpc.rpc_sync(trainer, set_shutdown_signal, args=())
    rpc.rpc_sync(self.remote_worker_name(), set_shutdown_signal, args=())