import ctypes
import sys
import pickle
import logging
from ..base import _LIB, check_call
from .base import create
def server_controller(cmd_id, cmd_body, _):
    """Server controler."""
    if not self.init_logginig:
        head = '%(asctime)-15s Server[' + str(self.kvstore.rank) + '] %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=head)
        self.init_logginig = True
    if cmd_id == 0:
        try:
            optimizer = pickle.loads(cmd_body)
        except:
            raise
        self.kvstore.set_optimizer(optimizer)
    else:
        print('server %d, unknown command (%d, %s)' % (self.kvstore.rank, cmd_id, cmd_body))