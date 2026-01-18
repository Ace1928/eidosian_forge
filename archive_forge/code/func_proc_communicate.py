import os
import sys
import subprocess
from urllib.parse import quote
from paste.util import converters
def proc_communicate(proc, stdin=None, stdout=None, stderr=None):
    """
    Run the given process, piping input/output/errors to the given
    file-like objects (which need not be actual file objects, unlike
    the arguments passed to Popen).  Wait for process to terminate.

    Note: this is taken from the posix version of
    subprocess.Popen.communicate, but made more general through the
    use of file-like objects.
    """
    read_set = []
    write_set = []
    input_buffer = b''
    trans_nl = proc.universal_newlines and hasattr(open, 'newlines')
    if proc.stdin:
        proc.stdin.flush()
        if input:
            write_set.append(proc.stdin)
        else:
            proc.stdin.close()
    else:
        assert stdin is None
    if proc.stdout:
        read_set.append(proc.stdout)
    else:
        assert stdout is None
    if proc.stderr:
        read_set.append(proc.stderr)
    else:
        assert stderr is None
    while read_set or write_set:
        rlist, wlist, xlist = select.select(read_set, write_set, [])
        if proc.stdin in wlist:
            next, input_buffer = (input_buffer, b'')
            next_len = 512 - len(next)
            if next_len:
                next += stdin.read(next_len)
            if not next:
                proc.stdin.close()
                write_set.remove(proc.stdin)
            else:
                bytes_written = os.write(proc.stdin.fileno(), next)
                if bytes_written < len(next):
                    input_buffer = next[bytes_written:]
        if proc.stdout in rlist:
            data = os.read(proc.stdout.fileno(), 1024)
            if data == b'':
                proc.stdout.close()
                read_set.remove(proc.stdout)
            if trans_nl:
                data = proc._translate_newlines(data)
            stdout.write(data)
        if proc.stderr in rlist:
            data = os.read(proc.stderr.fileno(), 1024)
            if data == b'':
                proc.stderr.close()
                read_set.remove(proc.stderr)
            if trans_nl:
                data = proc._translate_newlines(data)
            stderr.write(ensure_text(data))
    try:
        proc.wait()
    except OSError as e:
        if e.errno != 10:
            raise