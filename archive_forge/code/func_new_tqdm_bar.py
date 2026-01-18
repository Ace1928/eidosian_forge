from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def new_tqdm_bar(self, bar):
    """Create a new tqdm bar, possibly replacing an existing one."""
    if bar in self.tqdm_bars and self.tqdm_bars[bar] is not None:
        self.close_tqdm_bar(bar)
    infos = self.bars[bar]
    self.tqdm_bars[bar] = self.tqdm(total=infos['total'], desc=infos['title'], postfix=dict(now=troncate_string(str(infos['message']))), leave=self.leave_bars)