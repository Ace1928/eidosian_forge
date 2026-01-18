from __future__ import annotations
import importlib.metadata
def get_commit(tag: str, version: str) -> str:
    version_re = '[0-9.]+\\.dev[0-9]+\\+g([0-9a-f]{7,}|unknown)(?:\\.dirty)?'
    match = re.fullmatch(version_re, version)
    assert match is not None
    commit, = match.groups()
    return tag if commit == 'unknown' else commit