import os
from abc import ABC, abstractmethod
import torch.testing._internal.dist_utils
@property
def init_method(self):
    use_tcp_init = os.environ.get('RPC_INIT_WITH_TCP', None)
    if use_tcp_init == '1':
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        return f'tcp://{master_addr}:{master_port}'
    else:
        return self.file_init_method