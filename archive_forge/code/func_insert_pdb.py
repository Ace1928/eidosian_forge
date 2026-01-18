import torch.fx as fx
def insert_pdb(body):
    return ['import pdb; pdb.set_trace()\n', *body]