import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
def scan_cache_dir(cache_dir: Optional[Union[str, Path]]=None) -> HFCacheInfo:
    """Scan the entire HF cache-system and return a [`~HFCacheInfo`] structure.

    Use `scan_cache_dir` in order to programmatically scan your cache-system. The cache
    will be scanned repo by repo. If a repo is corrupted, a [`~CorruptedCacheException`]
    will be thrown internally but captured and returned in the [`~HFCacheInfo`]
    structure. Only valid repos get a proper report.

    ```py
    >>> from huggingface_hub import scan_cache_dir

    >>> hf_cache_info = scan_cache_dir()
    HFCacheInfo(
        size_on_disk=3398085269,
        repos=frozenset({
            CachedRepoInfo(
                repo_id='t5-small',
                repo_type='model',
                repo_path=PosixPath(...),
                size_on_disk=970726914,
                nb_files=11,
                revisions=frozenset({
                    CachedRevisionInfo(
                        commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                        size_on_disk=970726339,
                        snapshot_path=PosixPath(...),
                        files=frozenset({
                            CachedFileInfo(
                                file_name='config.json',
                                size_on_disk=1197
                                file_path=PosixPath(...),
                                blob_path=PosixPath(...),
                            ),
                            CachedFileInfo(...),
                            ...
                        }),
                    ),
                    CachedRevisionInfo(...),
                    ...
                }),
            ),
            CachedRepoInfo(...),
            ...
        }),
        warnings=[
            CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
            CorruptedCacheException(...),
            ...
        ],
    )
    ```

    You can also print a detailed report directly from the `huggingface-cli` using:
    ```text
    > huggingface-cli scan-cache
    REPO ID                     REPO TYPE SIZE ON DISK NB FILES REFS                LOCAL PATH
    --------------------------- --------- ------------ -------- ------------------- -------------------------------------------------------------------------
    glue                        dataset         116.3K       15 1.17.0, main, 2.4.0 /Users/lucain/.cache/huggingface/hub/datasets--glue
    google/fleurs               dataset          64.9M        6 main, refs/pr/1     /Users/lucain/.cache/huggingface/hub/datasets--google--fleurs
    Jean-Baptiste/camembert-ner model           441.0M        7 main                /Users/lucain/.cache/huggingface/hub/models--Jean-Baptiste--camembert-ner
    bert-base-cased             model             1.9G       13 main                /Users/lucain/.cache/huggingface/hub/models--bert-base-cased
    t5-base                     model            10.1K        3 main                /Users/lucain/.cache/huggingface/hub/models--t5-base
    t5-small                    model           970.7M       11 refs/pr/1, main     /Users/lucain/.cache/huggingface/hub/models--t5-small

    Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
    Got 1 warning(s) while scanning. Use -vvv to print details.
    ```

    Args:
        cache_dir (`str` or `Path`, `optional`):
            Cache directory to cache. Defaults to the default HF cache directory.

    <Tip warning={true}>

    Raises:

        `CacheNotFound`
          If the cache directory does not exist.

        [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          If the cache directory is a file, instead of a directory.

    </Tip>

    Returns: a [`~HFCacheInfo`] object.
    """
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise CacheNotFound(f'Cache directory not found: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.', cache_dir=cache_dir)
    if cache_dir.is_file():
        raise ValueError(f'Scan cache expects a directory but found a file: {cache_dir}. Please use `cache_dir` argument or set `HF_HUB_CACHE` environment variable.')
    repos: Set[CachedRepoInfo] = set()
    warnings: List[CorruptedCacheException] = []
    for repo_path in cache_dir.iterdir():
        if repo_path.name == '.locks':
            continue
        try:
            repos.add(_scan_cached_repo(repo_path))
        except CorruptedCacheException as e:
            warnings.append(e)
    return HFCacheInfo(repos=frozenset(repos), size_on_disk=sum((repo.size_on_disk for repo in repos)), warnings=warnings)