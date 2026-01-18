from __future__ import absolute_import, division, print_function
import errno
import os
import subprocess
import sys
import tempfile
import numpy as np
from ..utils import string_types
from ..audio.signal import Signal
def decode_to_pipe(infile, fmt='f32le', sample_rate=None, num_channels=1, skip=None, max_len=None, buf_size=-1, cmd='ffmpeg'):
    """
    Decode the given audio and return a file-like object for reading the
    samples, as well as a process object.

    Parameters
    ----------
    infile : str
        Name of the audio sound file to decode.
    fmt : {'f32le', 's16le'}, optional
        Format of the samples:
        - 'f32le' for float32, little-endian,
        - 's16le' for signed 16-bit int, little-endian.
    sample_rate : int, optional
        Sample rate to re-sample the signal to (if set) [Hz].
    num_channels : int, optional
        Number of channels to reduce the signal to.
    skip : float, optional
        Number of seconds to skip at beginning of file.
    max_len : float, optional
        Maximum length in seconds to decode.
    buf_size : int, optional
        Size of buffer for the file-like object:
        - '-1' means OS default (default),
        - '0' means unbuffered,
        - '1' means line-buffered, any other value is the buffer size in bytes.
    cmd : {'ffmpeg','avconv'}, optional
        Decoding command (defaults to ffmpeg, alternatively supports avconv).

    Returns
    -------
    pipe : file-like object
        File-like object for reading the decoded samples.
    proc : process object
        Process object for the decoding process.

    Notes
    -----
    To stop decoding the file, call close() on the returned file-like object,
    then call wait() on the returned process object.

    """
    if not isinstance(infile, (string_types, Signal)):
        raise ValueError('only file names or Signal instances are supported as `infile`, not %s.' % infile)
    call = _ffmpeg_call(infile, 'pipe:1', fmt, sample_rate, num_channels, skip, max_len, cmd)
    if isinstance(infile, Signal):
        proc = subprocess.Popen(call, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=buf_size)
    else:
        proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=buf_size)
    return (proc.stdout, proc)