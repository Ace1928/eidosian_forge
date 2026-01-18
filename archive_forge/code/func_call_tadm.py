import subprocess
import sys
from nltk.internals import find_binary
def call_tadm(args):
    """
    Call the ``tadm`` binary with the given arguments.
    """
    if isinstance(args, str):
        raise TypeError('args should be a list of strings')
    if _tadm_bin is None:
        config_tadm()
    cmd = [_tadm_bin] + args
    p = subprocess.Popen(cmd, stdout=sys.stdout)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print()
        print(stderr)
        raise OSError('tadm command failed!')