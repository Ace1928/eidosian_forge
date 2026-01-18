import logging
import comm.base_comm
import traitlets
import traitlets.config
from .comm import Comm
def comm_open(self, stream, ident, msg):
    """Handler for comm_open messages"""
    content = msg['content']
    comm_id = content['comm_id']
    target_name = content['target_name']
    f = self.targets.get(target_name, None)
    comm = Comm(comm_id=comm_id, primary=False, target_name=target_name, show_warning=False)
    self.register_comm(comm)
    if f is None:
        logger.error('No such comm target registered: %s', target_name)
    else:
        try:
            f(comm, msg)
            return
        except Exception:
            logger.error('Exception opening comm with target: %s', target_name, exc_info=True)
    try:
        comm.close()
    except Exception:
        logger.error('Could not close comm during `comm_open` failure\n                clean-up.  The comm may not have been opened yet.', exc_info=True)