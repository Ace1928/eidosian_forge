import errno
import os
import sys
import tempfile
from subprocess import PIPE, Popen
from .errors import BzrError, NoDiff3
from .textfile import check_text_path
def diff3(out_file, mine_path, older_path, yours_path):

    def add_label(args, label):
        args.extend(('-L', label))
    check_text_path(mine_path)
    check_text_path(older_path)
    check_text_path(yours_path)
    args = ['diff3', '-E', '--merge']
    add_label(args, 'TREE')
    add_label(args, 'ANCESTOR')
    add_label(args, 'MERGE-SOURCE')
    args.extend((mine_path, older_path, yours_path))
    try:
        output, stderr, status = write_to_cmd(args)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise NoDiff3
        else:
            raise
    if status not in (0, 1):
        raise Exception(stderr)
    with open(out_file, 'wb') as f:
        f.write(output)
    return status