import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import click
def install_powershell(*, prog_name: str, complete_var: str, shell: str) -> Path:
    subprocess.run([shell, '-Command', 'Set-ExecutionPolicy', 'Unrestricted', '-Scope', 'CurrentUser'])
    result = subprocess.run([shell, '-NoProfile', '-Command', 'echo', '$profile'], check=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        click.echo("Couldn't get PowerShell user profile", err=True)
        raise click.exceptions.Exit(result.returncode)
    path_str = ''
    if isinstance(result.stdout, str):
        path_str = result.stdout
    if isinstance(result.stdout, bytes):
        try:
            path_str = result.stdout.decode('windows-1252')
        except UnicodeDecodeError:
            try:
                path_str = result.stdout.decode('utf8')
            except UnicodeDecodeError:
                click.echo("Couldn't decode the path automatically", err=True)
                raise
    path_obj = Path(path_str.strip())
    parent_dir: Path = path_obj.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    script_content = get_completion_script(prog_name=prog_name, complete_var=complete_var, shell=shell)
    with path_obj.open(mode='a') as f:
        f.write(f'{script_content}\n')
    return path_obj