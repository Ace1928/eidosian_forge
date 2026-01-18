from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def dump_logs(self, filepath=None):
    if filepath is not None:
        with open(filepath, 'a') as f:
            f.write('\n'.join(self.logs))
    else:
        return '\n'.join(self.logs)