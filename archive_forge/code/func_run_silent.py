import os
import tempfile
import subprocess
from subprocess import PIPE
def run_silent(*command):
    subprocess.Popen(command, stdout=PIPE, stderr=PIPE).wait()