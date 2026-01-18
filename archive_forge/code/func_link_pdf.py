import os
import tempfile
import subprocess
from subprocess import PIPE
def link_pdf(peer_code):
    curr_dir = os.path.abspath(os.path.curdir)
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)
    ans = run_draw(peer_code, 'peer_code', 'link.mps', '--pen-size=4', '--disc-size=40')
    if len(ans):
        os.chdir(curr_dir)
        raise ValueError('draw failed: ' + ans + 'for ' + peer_code)
    run_silent('mpost', 'link.mps')
    run_silent('env', 'epstopdf', 'link.1')
    with open('link.pdf', 'rb') as f:
        data = f.read()
    os.chdir(curr_dir)
    os.system('rm -rf ' + tmp_dir)
    return data