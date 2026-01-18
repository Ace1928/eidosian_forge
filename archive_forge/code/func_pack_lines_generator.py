import pandas as pd
from .utils import series_to_line
def pack_lines_generator(self, pack_size):
    lines = []
    group_ids = []
    current_pack_size = 0
    lines_generator = self.lines_generator()
    for group_id, line in lines_generator:
        group_ids.append(group_id)
        lines.append(line)
        current_pack_size += 1
        if current_pack_size == pack_size:
            yield (group_ids, lines)
            lines = []
            group_ids = []
            current_pack_size = 0
    if current_pack_size != 0:
        yield (group_ids, lines)