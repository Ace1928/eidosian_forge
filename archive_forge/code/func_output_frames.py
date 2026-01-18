from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def output_frames(frames, reverse, limit):
    if limit == 0:
        return
    if reverse:
        assert isinstance(frames, list)
        frames = reversed(frames)
    if limit is not None:
        frames = list(frames)
        frames = frames[-limit:]
    for frame in frames:
        f, filename, line, fnname, text, index = frame
        text = text[0].strip() if text else ''
        if filename.startswith(cwd):
            filename = filename[len(cwd):]
        if filename.startswith(f'.{os.sep}'):
            filename = filename[2:]
        if _filelinefn:
            out.write(f'    {filename}:{line}:{fnname}(): {text}\n')
        else:
            out.write(f'    {fnname}(): {text}\n')