from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def freebsd_syms(libfilename: str, outfilename: str) -> None:
    output = call_tool('readelf', ['-d', libfilename])
    if not output:
        dummy_syms(outfilename)
        return
    result = [x for x in output.split('\n') if 'SONAME' in x]
    assert len(result) <= 1
    output = call_tool('nm', ['--dynamic', '--extern-only', '--defined-only', '--format=posix', libfilename])
    if not output:
        dummy_syms(outfilename)
        return
    result += [' '.join(x.split()[0:2]) for x in output.split('\n')]
    write_if_changed('\n'.join(result) + '\n', outfilename)