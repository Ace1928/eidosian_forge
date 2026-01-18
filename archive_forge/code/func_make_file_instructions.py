import copy
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
from .download.download_config import DownloadConfig
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import cached_path
def make_file_instructions(name: str, split_infos: List['SplitInfo'], instruction: Union[str, 'ReadInstruction'], filetype_suffix: Optional[str]=None, prefix_path: Optional[str]=None) -> FileInstructions:
    """Returns instructions of the split dict.

    Args:
        name (`str`): Name of the dataset.
        split_infos (`list` of `[SplitInfo]`): Dataset splits information.
        instruction ([`ReadInstruction`] or `str`): Reading instruction for a dataset.
        filetype_suffix (`str`, *optional*): Suffix of dataset files, e.g. 'arrow' or 'parquet'.
        prefix_path (`str`, *optional*): Prefix of dataset files, e.g. directory name.

    Returns:
        [`FileInstructions`]
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected str 'name', but got: {type(name).__name__}")
    elif not name:
        raise ValueError("Expected non-empty str 'name'")
    name2len = {info.name: info.num_examples for info in split_infos}
    name2shard_lengths = {info.name: info.shard_lengths for info in split_infos}
    name2filenames = {info.name: filenames_for_dataset_split(path=prefix_path, dataset_name=name, split=info.name, filetype_suffix=filetype_suffix, shard_lengths=name2shard_lengths[info.name]) for info in split_infos}
    if not isinstance(instruction, ReadInstruction):
        instruction = ReadInstruction.from_spec(instruction)
    absolute_instructions = instruction.to_absolute(name2len)
    file_instructions = []
    num_examples = 0
    for abs_instr in absolute_instructions:
        split_length = name2len[abs_instr.splitname]
        filenames = name2filenames[abs_instr.splitname]
        shard_lengths = name2shard_lengths[abs_instr.splitname]
        from_ = 0 if abs_instr.from_ is None else abs_instr.from_
        to = split_length if abs_instr.to is None else abs_instr.to
        if shard_lengths is None:
            for filename in filenames:
                take = to - from_
                if take == 0:
                    continue
                num_examples += take
                file_instructions.append({'filename': filename, 'skip': from_, 'take': take})
        else:
            index_start = 0
            index_end = 0
            for filename, shard_length in zip(filenames, shard_lengths):
                index_end += shard_length
                if from_ < index_end and to > index_start:
                    skip = from_ - index_start if from_ > index_start else 0
                    take = to - index_start - skip if to < index_end else -1
                    if take == 0:
                        continue
                    file_instructions.append({'filename': filename, 'skip': skip, 'take': take})
                    num_examples += shard_length - skip if take == -1 else take
                index_start += shard_length
    return FileInstructions(num_examples=num_examples, file_instructions=file_instructions)