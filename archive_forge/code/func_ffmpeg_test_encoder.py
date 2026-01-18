import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from functools import lru_cache
from ._parsing import LogCatcher, cvsecs, parse_ffmpeg_header
from ._utils import _popen_kwargs, get_ffmpeg_exe, logger
def ffmpeg_test_encoder(encoder):
    cmd = [get_ffmpeg_exe(), '-hide_banner', '-f', 'lavfi', '-i', 'nullsrc=s=256x256:d=8', '-vcodec', encoder, '-f', 'null', '-']
    p = subprocess.run(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode == 0