import os
import re
import shutil
import tempfile
from datetime import datetime
from time import time
from typing import List, Optional
from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, Method
from cmdstanpy.utils import get_logger
def get_err_msgs(self) -> str:
    """Checks console messages for each CmdStan run."""
    msgs = []
    for i in range(self._num_procs):
        if os.path.exists(self._stdout_files[i]) and os.stat(self._stdout_files[i]).st_size > 0:
            if self._args.method == Method.OPTIMIZE:
                msgs.append('console log output:\n')
                with open(self._stdout_files[0], 'r') as fd:
                    msgs.append(fd.read())
            else:
                with open(self._stdout_files[i], 'r') as fd:
                    contents = fd.read()
                    pat = re.compile('^E[rx].*$', re.M)
                    errors = re.findall(pat, contents)
                    if len(errors) > 0:
                        msgs.append('\n\t'.join(errors))
    return '\n'.join(msgs)