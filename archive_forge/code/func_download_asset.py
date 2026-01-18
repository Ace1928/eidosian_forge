import hashlib
import logging
from os import PathLike
from pathlib import Path
from typing import Union
import torch
from torchaudio._internal import download_url_to_file
def download_asset(key: str, hash: str='', path: Union[str, PathLike]='', *, progress: bool=True) -> str:
    """Download and store torchaudio assets to local file system.

    If a file exists at the download path, then that path is returned with or without
    hash validation.

    Args:
        key (str): The asset identifier.
        hash (str, optional):
            The value of SHA256 hash of the asset. If provided, it is used to verify
            the downloaded / cached object. If not provided, then no hash validation
            is performed. This means if a file exists at the download path, then the path
            is returned as-is without verifying the identity of the file.
        path (path-like object, optional):
            By default, the downloaded asset is saved in a directory under
            :py:func:`torch.hub.get_dir` and intermediate directories based on the given `key`
            are created.
            This argument can be used to overwrite the target location.
            When this argument is provided, all the intermediate directories have to be
            created beforehand.
        progress (bool): Whether to show progress bar for downloading. Default: ``True``.

    Note:
        Currently the valid key values are the route on ``download.pytorch.org/torchaudio``,
        but this is an implementation detail.

    Returns:
        str: The path to the asset on the local file system.
    """
    path = path or _get_local_path(key)
    if path.exists():
        _LG.info('The local file (%s) exists. Skipping the download.', path)
    else:
        _LG.info('Downloading %s to %s', key, path)
        _download(key, path, progress=progress)
    if hash:
        _LG.info('Verifying the hash value.')
        digest = _get_hash(path, hash)
        if digest != hash:
            raise ValueError(f"The hash value of the downloaded file ({path}), '{digest}' does not match the provided hash value, '{hash}'.")
        _LG.info('Hash validated.')
    return str(path)