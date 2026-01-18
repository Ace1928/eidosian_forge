import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def populate_rows_and_columns(self) -> Tuple[Tuple[_Row, ...], Tuple[_Column, ...]]:
    rows: List[_Row] = []
    columns: List[_Column] = []
    ordered_results: List[List[Optional[common.Measurement]]] = [[None for _ in self.column_keys] for _ in self.row_keys]
    row_position = {key: i for i, key in enumerate(self.row_keys)}
    col_position = {key: i for i, key in enumerate(self.column_keys)}
    for r in self.results:
        i = row_position[self.row_fn(r)]
        j = col_position[self.col_fn(r)]
        ordered_results[i][j] = r
    unique_envs = {r.env for r in self.results}
    render_env = len(unique_envs) > 1
    env_str_len = max((len(i) for i in unique_envs)) if render_env else 0
    row_name_str_len = max((len(r.as_row_name) for r in self.results))
    prior_num_threads = -1
    prior_env = ''
    row_group = -1
    rows_by_group: List[List[List[Optional[common.Measurement]]]] = []
    for (num_threads, env, _), row in zip(self.row_keys, ordered_results):
        thread_transition = num_threads != prior_num_threads
        if thread_transition:
            prior_num_threads = num_threads
            prior_env = ''
            row_group += 1
            rows_by_group.append([])
        rows.append(_Row(results=row, row_group=row_group, render_env=render_env and env != prior_env, env_str_len=env_str_len, row_name_str_len=row_name_str_len, time_scale=self.time_scale, colorize=self._colorize, num_threads=num_threads if thread_transition else None))
        rows_by_group[-1].append(row)
        prior_env = env
    for i in range(len(self.column_keys)):
        grouped_results = [tuple((row[i] for row in g)) for g in rows_by_group]
        column = _Column(grouped_results=grouped_results, time_scale=self.time_scale, time_unit=self.time_unit, trim_significant_figures=self._trim_significant_figures, highlight_warnings=self._highlight_warnings)
        columns.append(column)
    rows_tuple, columns_tuple = (tuple(rows), tuple(columns))
    for ri in rows_tuple:
        ri.register_columns(columns_tuple)
    return (rows_tuple, columns_tuple)