import csv
import glob
import os
import warnings
from contextlib import ExitStack
from typing import List, Tuple
import fsspec
import pandas
import pandas._libs.lib as lib
from pandas.io.common import is_fsspec_url, is_url, stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.csv_dispatcher import CSVDispatcher
@classmethod
def partitioned_file(cls, files, fnames: List[str], num_partitions: int=None, nrows: int=None, skiprows: int=None, skip_header: int=None, quotechar: bytes=b'"', is_quoting: bool=True) -> List[List[Tuple[str, int, int]]]:
    """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        files : file or list of files
            File(s) to be partitioned.
        fnames : str or list of str
            File name(s) to be partitioned.
        num_partitions : int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        nrows : int, optional
            Number of rows of file to read.
        skiprows : int, optional
            Specifies rows to skip.
        skip_header : int, optional
            Specifies header rows to skip.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.

        Returns
        -------
        list
            List, where each element of the list is a list of tuples. The inner lists
            of tuples contains the data file name of the chunk, chunk start offset, and
            chunk end offsets for its corresponding file.

        Notes
        -----
        The logic gets really complicated if we try to use the `TextFileDispatcher.partitioned_file`.
        """
    if type(files) is not list:
        files = [files]
    if num_partitions is None:
        num_partitions = NPartitions.get()
    file_sizes = [cls.file_size(f) for f in files]
    partition_size = max(1, num_partitions, (nrows if nrows else sum(file_sizes)) // num_partitions)
    result = []
    split_result = []
    split_size = 0
    read_rows_counter = 0
    for f, fname, f_size in zip(files, fnames, file_sizes):
        if skiprows or skip_header:
            skip_amount = (skiprows if skiprows else 0) + (skip_header if skip_header else 0)
            outside_quotes, read_rows = cls._read_rows(f, nrows=skip_amount, quotechar=quotechar, is_quoting=is_quoting)
            if skiprows:
                skiprows -= read_rows
                if skiprows > 0:
                    continue
        start = f.tell()
        while f.tell() < f_size:
            if split_size >= partition_size:
                result.append(split_result)
                split_result = []
                split_size = 0
            read_size = partition_size - split_size
            if nrows:
                if read_rows_counter >= nrows:
                    if len(split_result) > 0:
                        result.append(split_result)
                    return result
                elif read_rows_counter + read_size > nrows:
                    read_size = nrows - read_rows_counter
                outside_quotes, read_rows = cls._read_rows(f, nrows=read_size, quotechar=quotechar, is_quoting=is_quoting)
                split_size += read_rows
                read_rows_counter += read_rows
            else:
                outside_quotes = cls.offset(f, offset_size=read_size, quotechar=quotechar, is_quoting=is_quoting)
            split_result.append((fname, start, f.tell()))
            split_size += f.tell() - start
            start = f.tell()
            if is_quoting and (not outside_quotes):
                warnings.warn('File has mismatched quotes')
    if len(split_result) > 0:
        result.append(split_result)
    return result