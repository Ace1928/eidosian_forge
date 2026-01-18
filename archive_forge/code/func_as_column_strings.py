import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def as_column_strings(self):
    concrete_results = [r for r in self._results if r is not None]
    env = f'({concrete_results[0].env})' if self._render_env else ''
    env = env.ljust(self._env_str_len + 4)
    output = ['  ' + env + concrete_results[0].as_row_name]
    for m, col in zip(self._results, self._columns or ()):
        if m is None:
            output.append(col.num_to_str(None, 1, None))
        else:
            output.append(col.num_to_str(m.median / self._time_scale, m.significant_figures, m.iqr / m.median if m.has_warnings else None))
    return output