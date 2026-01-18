from fnmatch import fnmatch
import os
from pathlib import Path
import re
import subprocess
import pytest
import cartopy
def test_license_headers(self):
    exclude_patterns = ('build/*', 'dist/*', 'docs/build/*', 'docs/source/gallery/*', 'examples/*', 'lib/cartopy/_version.py')
    try:
        tracked_files = self.list_tracked_files()
    except ValueError as e:
        return pytest.skip(f'cartopy installation did not look like a git repo: {e}')
    failed = []
    for fname in sorted(tracked_files):
        full_fname = REPO_DIR / fname
        ext = full_fname.suffix
        if ext in ('.py', '.pyx', '.c', '.cpp', '.h') and full_fname.is_file() and (not any((fnmatch(fname, pat) for pat in exclude_patterns))):
            if full_fname.stat().st_size == 0:
                continue
            with open(full_fname, encoding='utf-8') as fh:
                content = fh.read()
            if not bool(LICENSE_RE.match(content)):
                failed.append(full_fname)
    assert failed == [], 'There were license header failures.'