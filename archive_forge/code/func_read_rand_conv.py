import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
def read_rand_conv(self):
    idx = random.choice(range(len(self)))
    self.read_conv_idx(idx)