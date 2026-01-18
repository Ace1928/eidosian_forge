import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=['SCRIPT_REMOTE_CALL'])
def test_builtin_remote_message_dropped_timeout_to_self(self):
    func = torch.add
    args = (torch.tensor(1), torch.tensor(1))
    self._test_remote_message_dropped_timeout(func, args, dst=0)