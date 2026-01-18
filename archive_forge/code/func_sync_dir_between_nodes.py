import fnmatch
import io
import os
import shutil
import tarfile
from typing import Optional, Tuple, Dict, Generator, Union, List
import ray
from ray.util.annotations import DeveloperAPI
from ray.air._internal.filelock import TempFileLock
from ray.air.util.node import _get_node_id_from_node_ip, _force_on_node
@DeveloperAPI
def sync_dir_between_nodes(source_ip: str, source_path: str, target_ip: str, target_path: str, force_all: bool=False, exclude: Optional[List]=None, chunk_size_bytes: int=_DEFAULT_CHUNK_SIZE_BYTES, max_size_bytes: Optional[int]=_DEFAULT_MAX_SIZE_BYTES, return_futures: bool=False) -> Union[None, Tuple[ray.ObjectRef, ray.ActorID, ray.ObjectRef], Tuple[ray.ObjectRef, None, None]]:
    """Synchronize directory on source node to directory on target node.

    Per default, this function will collect information about already existing
    files in the target directory. Only files that differ in either mtime or
    filesize will be transferred, unless ``force_all=True``.

    If ``source_ip==target_ip``, shutil will be used to copy the directory. Otherwise,
    the directory will be packed and sent through the Ray Object Store to the target
    node.

    Args:
        source_ip: IP of source node.
        source_path: Path to directory on source node.
        target_ip: IP of target node.
        target_path: Path to directory on target node.
        force_all: If True, all files will be transferred (not just differing files).
            Ignored if ``source_ip==target_ip``.
        exclude: Pattern of files to exclude, e.g.
            ``["*/checkpoint_*]`` to exclude trial checkpoints.
        chunk_size_bytes: Chunk size for data transfer. Ignored if
            ``source_ip==target_ip``.
        max_size_bytes: If packed data exceeds this value, raise an error before
            transfer. If ``None``, no limit is enforced. Ignored if
            ``source_ip==target_ip``.
        return_futures: If True, returns a tuple of the unpack future,
            the pack actor, and the files_stats future. If False (default) will
            block until synchronization finished and return None.

    Returns:
        None, or Tuple of unpack future, pack actor, and files_stats future.
        If ``source_ip==target_ip``, pack actor and files_stats future will be None.

    """
    if source_ip != target_ip:
        return _sync_dir_between_different_nodes(source_ip=source_ip, source_path=source_path, target_ip=target_ip, target_path=target_path, force_all=force_all, exclude=exclude, chunk_size_bytes=chunk_size_bytes, max_size_bytes=max_size_bytes, return_futures=return_futures)
    elif source_path != target_path:
        ret = _sync_dir_on_same_node(ip=source_ip, source_path=source_path, target_path=target_path, exclude=exclude, return_futures=return_futures)
        if return_futures:
            return (ret, None, None)
        return ret