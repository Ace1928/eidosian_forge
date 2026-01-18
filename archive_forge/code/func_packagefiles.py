from __future__ import annotations
from dataclasses import dataclass, InitVar
import os, subprocess
import argparse
import asyncio
import threading
import copy
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import typing as T
import tarfile
import zipfile
from . import mlog
from .ast import IntrospectionInterpreter
from .mesonlib import quiet_git, GitException, Popen_safe, MesonException, windows_proof_rmtree
from .wrap.wrap import (Resolver, WrapException, ALL_TYPES,
def packagefiles(self) -> bool:
    options = T.cast('PackagefilesArguments', self.options)
    if options.apply and options.save:
        print('error: --apply and --save are mutually exclusive')
        return False
    if options.apply:
        self.log(f'Re-applying patchfiles overlay for {self.wrap.name}...')
        if not os.path.isdir(self.repo_dir):
            self.log('  -> Not downloaded yet')
            return True
        self.wrap_resolver.apply_patch(self.wrap.name)
        return True
    if options.save:
        if 'patch_directory' not in self.wrap.values:
            mlog.error('can only save packagefiles to patch_directory')
            return False
        if 'source_filename' not in self.wrap.values:
            mlog.error('can only save packagefiles from a [wrap-file]')
            return False
        archive_path = Path(self.wrap_resolver.cachedir, self.wrap.values['source_filename'])
        lead_directory_missing = bool(self.wrap.values.get('lead_directory_missing', False))
        directory = Path(self.repo_dir)
        packagefiles = Path(self.wrap.filesdir, self.wrap.values['patch_directory'])
        base_path = directory if lead_directory_missing else directory.parent
        archive_files = read_archive_files(archive_path, base_path)
        directory_files = set(directory.glob('**/*'))
        self.log(f'Saving {self.wrap.name} to {packagefiles}...')
        shutil.rmtree(packagefiles)
        for src_path in directory_files - archive_files:
            if not src_path.is_file():
                continue
            rel_path = src_path.relative_to(directory)
            dst_path = packagefiles / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
    return True