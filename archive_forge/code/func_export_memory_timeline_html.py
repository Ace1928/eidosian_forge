import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def export_memory_timeline_html(self, path, device, figsize=(20, 12), title=None) -> None:
    """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
    import importlib.util
    matplotlib_spec = importlib.util.find_spec('matplotlib')
    if matplotlib_spec is None:
        print('export_memory_timeline_html failed because matplotlib was not found.')
        return
    from base64 import b64encode
    from os import remove
    from tempfile import NamedTemporaryFile
    import matplotlib.pyplot as plt
    import numpy as np
    mt = self._coalesce_timeline(device)
    times, sizes = (np.array(mt[0]), np.array(mt[1]))
    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / 1024 ** 3
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()
    fig = plt.figure(figsize=figsize, dpi=80)
    axes = fig.gca()
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        axes.fill_between(times / 1000.0, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7)
    fig.legend(['Unknown' if i is None else i.name for i in _CATEGORY_TO_COLORS])
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Memory (GB)')
    title = '\n\n'.join(([title] if title else []) + [f'Max memory allocated: {max_memory_allocated / 10 ** 9:.2f} GB \nMax memory reserved: {max_memory_reserved / 10 ** 9:.2f} GB'])
    axes.set_title(title)
    tmpfile = NamedTemporaryFile('wb', suffix='.png', delete=False)
    tmpfile.close()
    fig.savefig(tmpfile.name, format='png')
    with open(tmpfile.name, 'rb') as tmp:
        encoded = b64encode(tmp.read()).decode('utf-8')
        html = f"""<html>\n<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>\n<body>\n  <img src='data:image/png;base64,{encoded}'>\n</body>\n</html>"""
        with open(path, 'w') as f:
            f.write(html)
    remove(tmpfile.name)