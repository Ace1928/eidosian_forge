import torch.distributed.rpc as rpc
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import (
def get_shutdown_error_regex(self):
    error_regexes = ['.*']
    return '|'.join([f'({error_str})' for error_str in error_regexes])