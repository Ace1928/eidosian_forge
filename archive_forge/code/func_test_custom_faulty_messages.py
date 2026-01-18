import torch
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@dist_init(faulty_messages=['RREF_FORK_REQUEST', 'RREF_CHILD_ACCEPT'])
def test_custom_faulty_messages(self):
    self.assertEqual({'RREF_FORK_REQUEST', 'RREF_CHILD_ACCEPT'}, set(self.rpc_backend_options.messages_to_fail))