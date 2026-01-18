from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def bars_callback(self, bar, attr, value, old_value):
    if bar not in self.tqdm_bars or self.tqdm_bars[bar] is None:
        self.new_tqdm_bar(bar)
    if attr == 'index':
        if value >= old_value:
            total = self.bars[bar]['total']
            if total and value >= total:
                self.close_tqdm_bar(bar)
            else:
                self.tqdm_bars[bar].update(value - old_value)
        else:
            self.new_tqdm_bar(bar)
            self.tqdm_bars[bar].update(value + 1)
    elif attr == 'message':
        self.tqdm_bars[bar].set_postfix(now=troncate_string(str(value)))
        self.tqdm_bars[bar].update(0)