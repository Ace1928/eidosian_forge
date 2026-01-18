from __future__ import annotations
from typing import List, Optional, Union, Any, Set
import os
import glob
import json
from pathlib import Path
from pkg_resources import resource_filename
import pooch
from .exceptions import ParameterError
def example_info(key: str) -> None:
    """Display licensing and metadata information for the given example recording.

    The first time an example is requested, it will be downloaded from
    the remote repository over HTTPS.
    All subsequent requests will use a locally cached copy of the recording.

    For a list of examples (and their keys), see `librosa.util.list_examples`.

    By default, local files will be cached in the directory given by
    `pooch.os_cache('librosa')`.  You can override this by setting
    an environment variable ``LIBROSA_DATA_DIR`` prior to importing librosa.

    Parameters
    ----------
    key : str
        The identifier for the recording (see `list_examples`)

    See Also
    --------
    librosa.util.example
    librosa.util.list_examples
    pooch.os_cache
    """
    if key not in __TRACKMAP:
        raise ParameterError(f'Unknown example key: {key}')
    license_file = __GOODBOY.fetch(__TRACKMAP[key]['path'] + '.txt')
    with open(license_file, 'r') as fdesc:
        print(f'{key:10s}\t{__TRACKMAP[key]['desc']:s}')
        print('-' * 68)
        for line in fdesc:
            print(line)