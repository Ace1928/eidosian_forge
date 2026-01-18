import itertools
import os
import re
def filenames_for_dataset_split(path, dataset_name, split, filetype_suffix=None, shard_lengths=None):
    prefix = filename_prefix_for_split(dataset_name, split)
    prefix = os.path.join(path, prefix)
    if shard_lengths:
        num_shards = len(shard_lengths)
        filenames = [f'{prefix}-{shard_id:05d}-of-{num_shards:05d}' for shard_id in range(num_shards)]
        if filetype_suffix:
            filenames = [filename + f'.{filetype_suffix}' for filename in filenames]
        return filenames
    else:
        filename = prefix
        if filetype_suffix:
            filename += f'.{filetype_suffix}'
        return [filename]