import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
def example_kmeans_test(A, numCenter, numIter, init_centroids):
    centroids = init_centroids
    N, D = A.shape
    for l in range(numIter):
        dist = np.array([[math.sqrt(np.sum((A[i, :] - centroids[j, :]) ** 2)) for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i, :].argmin() for i in range(N)])
        centroids = np.array([[np.sum(A[labels == i, j]) / np.sum(labels == i) for j in range(D)] for i in range(numCenter)])
    return centroids