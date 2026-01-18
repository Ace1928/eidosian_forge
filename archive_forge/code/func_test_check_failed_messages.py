import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(messages_to_delay={})
def test_check_failed_messages(self):
    if self.rank == 0:
        dst_worker_b = worker_name((self.rank + 1) % self.world_size)
        dst_worker_c = worker_name((self.rank + 2) % self.world_size)
        rref = rpc.remote(dst_worker_b, torch.add, args=(torch.ones(2, 2), torch.ones(2, 2)))
        rpc.remote(dst_worker_c, add_rref_to_value, args=(rref, torch.ones(2, 2)))
        self.assertEqual(rref.to_here(), torch.add(torch.ones(2, 2), torch.ones(2, 2)))
    _delete_all_user_and_unforked_owner_rrefs()