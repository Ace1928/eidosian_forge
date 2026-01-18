import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
def check_env_var_off(self, env):
    src = 'if 1:\n        from numba import njit\n        import numpy as np\n        from numba.core.runtime import rtsys, _nrt_python\n\n        @njit\n        def foo():\n            return np.arange(10)[0]\n\n        assert _nrt_python.memsys_stats_enabled() == False\n        try:\n            rtsys.get_allocation_stats()\n        except RuntimeError as e:\n            assert "NRT stats are disabled." in str(e)\n        '
    run_in_subprocess(src, env=env)