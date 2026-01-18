import re
import sys
import time
from functools import partial, wraps
from typing import Tuple
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info
from torch.testing._internal.common_utils import FILE_SCHEMA, TEST_WITH_TSAN
@wraps(old_test_method)
def new_test_method(self, *arg, **kwargs):
    import torch.distributed.rpc.api as api
    api._ignore_rref_leak = False
    self.worker_id = self.rank
    self.setup_fault_injection(faulty_messages, messages_to_delay)
    rpc_backend_options = self.rpc_backend_options
    if setup_rpc:
        if TEST_WITH_TSAN:
            rpc_backend_options.rpc_timeout = rpc.constants.DEFAULT_RPC_TIMEOUT_SEC * 5
            rpc.constants.DEFAULT_SHUTDOWN_TIMEOUT = 60
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)
    return_value = old_test_method(self, *arg, **kwargs)
    if setup_rpc:
        rpc.shutdown(graceful=clean_shutdown)
    return return_value