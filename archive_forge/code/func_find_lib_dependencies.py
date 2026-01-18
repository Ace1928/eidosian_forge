import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
def find_lib_dependencies(llvm_readobj: Path, lib_path: Path, used_dependencies: Set[str]=None, dry_run: bool=False):
    """
    Find all the Qt dependencies of a library using llvm_readobj
    """
    if lib_path.name in used_dependencies:
        return
    command = [str(llvm_readobj), '--needed-libs', str(lib_path)]
    _, output = run_command(command=command, dry_run=dry_run, fetch_output=True)
    dependencies = set()
    neededlibraries_found = False
    for line in output.splitlines():
        line = line.decode('utf-8').lstrip()
        if line.startswith('NeededLibraries') and (not neededlibraries_found):
            neededlibraries_found = True
        if neededlibraries_found and line.startswith('libQt'):
            dependencies.add(line)
            used_dependencies.add(line)
            dependent_lib_path = lib_path.parent / line
            find_lib_dependencies(llvm_readobj, dependent_lib_path, used_dependencies, dry_run)
    if dependencies:
        logging.info(f'[DEPLOY] Following dependencies found for {lib_path.stem}: {dependencies}')
    else:
        logging.info(f'[DEPLOY] No Qt dependencies found for {lib_path.stem}')