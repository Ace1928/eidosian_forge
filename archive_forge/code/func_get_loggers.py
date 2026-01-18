import itertools
import logging
from torch.hub import _Faketqdm, tqdm
def get_loggers():
    return [logging.getLogger('torch.fx.experimental.symbolic_shapes'), logging.getLogger('torch._dynamo'), logging.getLogger('torch._inductor')]