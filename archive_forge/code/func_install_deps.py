import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Set
def install_deps(deps: List[str], failed: Optional[Set[str]]=None, extra_index: Optional[str]=None, opts: Optional[List[str]]=None) -> Optional[Set[str]]:
    """Install pip dependencies.

    Arguments:
        deps {List[str]} -- List of dependencies to install
        failed (set, None): The libraries that failed to install

    Returns:
        deps (str[], None): The dependencies that failed to install
    """
    try:
        clean_deps = [d.split('@')[-1].strip() if '@' in d else d for d in deps]
        index_args = ['--extra-index-url', extra_index] if extra_index else []
        print('installing {}...'.format(', '.join(clean_deps)))
        opts = opts or []
        args = ['pip', 'install'] + opts + clean_deps + index_args
        sys.stdout.flush()
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        return failed
    except subprocess.CalledProcessError as e:
        if failed is None:
            failed = set()
        num_failed = len(failed)
        current_pkg = None
        for line in e.output.decode('utf8').splitlines():
            current_pkg = get_current_package(line, clean_deps, current_pkg)
            if 'error: subprocess-exited-with-error' in line:
                if current_pkg is not None:
                    failed.add(current_pkg)
            elif line.startswith('ERROR:'):
                clean_dep = find_package_in_error_string(clean_deps, line)
                if clean_dep is not None:
                    if clean_dep in deps:
                        failed.add(clean_dep)
                    else:
                        for d in deps:
                            if clean_dep in d:
                                failed.add(d.replace(' ', ''))
                                break
        if len(set(clean_deps) - failed) == 0:
            return failed
        elif len(failed) > num_failed:
            return install_deps(list(set(clean_deps) - failed), failed, extra_index=extra_index, opts=opts)
        else:
            return failed