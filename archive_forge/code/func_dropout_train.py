import torch
import torch.nn.functional as F
def dropout_train(x):
    return F.dropout(x, p=0.5, training=True, inplace=inplace)