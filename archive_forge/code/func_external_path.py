from __future__ import annotations # isort:skip
import hashlib
import json
from os.path import splitext
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import urljoin
from urllib.request import urlopen
def external_path(file_name: str) -> Path:
    data_dir = external_data_dir()
    file_path = data_dir / file_name
    if not file_path.exists() or not file_path.is_file():
        raise RuntimeError(f'Could not locate external data file {file_path}. Please execute bokeh.sampledata.download()')
    with open(file_path, 'rb') as file:
        meta = metadata()
        known_md5 = meta.get(file_name) or meta.get(f'{file_name}.zip') or meta.get(f'{splitext(file_name)[0]}.zip')
        if known_md5 is None:
            raise RuntimeError(f'Unknown external data file {file_name}')
        local_md5 = hashlib.md5(file.read()).hexdigest()
        if known_md5 != local_md5:
            raise RuntimeError(f'External data file {file_path} is outdated. Please execute bokeh.sampledata.download()')
    return file_path