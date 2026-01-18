import sys
from IPython import get_ipython
import comm
def create_comm(*args, **kwargs):
    if requires_ipykernel_shim():
        from ipykernel.comm import Comm
        return Comm(*args, **kwargs)
    else:
        return comm.create_comm(*args, **kwargs)