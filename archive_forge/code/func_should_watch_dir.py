from __future__ import annotations
import logging
import warnings
from pathlib import Path
from socket import socket
from typing import TYPE_CHECKING, Callable
from watchgod import DefaultWatcher
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
def should_watch_dir(self, entry: DirEntry) -> bool:
    cached_result = self.watched_dirs.get(entry.path)
    if cached_result is not None:
        return cached_result
    entry_path = Path(entry)
    if entry_path in self.dirs_excludes:
        self.watched_dirs[entry.path] = False
        return False
    for exclude_pattern in self.excludes:
        if entry_path.match(exclude_pattern):
            is_watched = False
            if entry_path in self.dirs_includes:
                is_watched = True
            for directory in self.dirs_includes:
                if directory in entry_path.parents:
                    is_watched = True
            if is_watched:
                logger.debug("WatchGodReload detected a new excluded dir '%s' in '%s'; Adding to exclude list.", entry_path.relative_to(self.resolved_root), str(self.resolved_root))
            self.watched_dirs[entry.path] = False
            self.dirs_excludes.add(entry_path)
            return False
    if entry_path in self.dirs_includes:
        self.watched_dirs[entry.path] = True
        return True
    for directory in self.dirs_includes:
        if directory in entry_path.parents:
            self.watched_dirs[entry.path] = True
            return True
    for include_pattern in self.includes:
        if entry_path.match(include_pattern):
            logger.info("WatchGodReload detected a new reload dir '%s' in '%s'; Adding to watch list.", str(entry_path.relative_to(self.resolved_root)), str(self.resolved_root))
            self.dirs_includes.add(entry_path)
            self.watched_dirs[entry.path] = True
            return True
    self.watched_dirs[entry.path] = False
    return False