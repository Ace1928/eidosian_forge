import platform
import sys
import os
import re
import shutil
import warnings
import traceback
import llvmlite.binding as ll
def process_environ(self, environ):

    def _readenv(name, ctor, default):
        value = environ.get(name)
        if value is None:
            return default() if callable(default) else default
        try:
            return ctor(value)
        except Exception:
            warnings.warn(f"Environment variable '{name}' is defined but its associated value '{value}' could not be parsed.\nThe parse failed with exception:\n{traceback.format_exc()}", RuntimeWarning)
            return default

    def optional_str(x):
        return str(x) if x is not None else None
    USE_RVSDG_FRONTEND = _readenv('NUMBA_USE_RVSDG_FRONTEND', int, 0)
    DEVELOPER_MODE = _readenv('NUMBA_DEVELOPER_MODE', int, 0)
    DISABLE_PERFORMANCE_WARNINGS = _readenv('NUMBA_DISABLE_PERFORMANCE_WARNINGS', int, 0)
    FULL_TRACEBACKS = _readenv('NUMBA_FULL_TRACEBACKS', int, DEVELOPER_MODE)
    SHOW_HELP = _readenv('NUMBA_SHOW_HELP', int, 0)
    COLOR_SCHEME = _readenv('NUMBA_COLOR_SCHEME', str, 'no_color')
    BOUNDSCHECK = _readenv('NUMBA_BOUNDSCHECK', int, None)
    ALWAYS_WARN_UNINIT_VAR = _readenv('NUMBA_ALWAYS_WARN_UNINIT_VAR', int, 0)
    CUDA_LOW_OCCUPANCY_WARNINGS = _readenv('NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS', int, 1)
    CUDA_USE_NVIDIA_BINDING = _readenv('NUMBA_CUDA_USE_NVIDIA_BINDING', int, 0)
    DEBUG = _readenv('NUMBA_DEBUG', int, 0)
    DEBUG_PRINT_AFTER = _readenv('NUMBA_DEBUG_PRINT_AFTER', str, 'none')
    DEBUG_PRINT_BEFORE = _readenv('NUMBA_DEBUG_PRINT_BEFORE', str, 'none')
    DEBUG_PRINT_WRAP = _readenv('NUMBA_DEBUG_PRINT_WRAP', str, 'none')
    HIGHLIGHT_DUMPS = _readenv('NUMBA_HIGHLIGHT_DUMPS', int, 0)
    DEBUG_JIT = _readenv('NUMBA_DEBUG_JIT', int, 0)
    DEBUG_FRONTEND = _readenv('NUMBA_DEBUG_FRONTEND', int, 0)
    DEBUG_NRT = _readenv('NUMBA_DEBUG_NRT', int, 0)
    NRT_STATS = _readenv('NUMBA_NRT_STATS', int, 0)
    FUNCTION_CACHE_SIZE = _readenv('NUMBA_FUNCTION_CACHE_SIZE', int, 128)
    PARFOR_MAX_TUPLE_SIZE = _readenv('NUMBA_PARFOR_MAX_TUPLE_SIZE', int, 100)
    DEBUG_CACHE = _readenv('NUMBA_DEBUG_CACHE', int, DEBUG)
    CACHE_DIR = _readenv('NUMBA_CACHE_DIR', str, '')
    TRACE = _readenv('NUMBA_TRACE', int, 0)
    CHROME_TRACE = _readenv('NUMBA_CHROME_TRACE', str, '')
    DEBUG_TYPEINFER = _readenv('NUMBA_DEBUG_TYPEINFER', int, 0)
    CPU_NAME = _readenv('NUMBA_CPU_NAME', optional_str, None)
    CPU_FEATURES = _readenv('NUMBA_CPU_FEATURES', optional_str, '' if str(CPU_NAME).lower() == 'generic' else None)
    OPT = _readenv('NUMBA_OPT', _process_opt_level, _OptLevel(3))
    DUMP_BYTECODE = _readenv('NUMBA_DUMP_BYTECODE', int, DEBUG_FRONTEND)
    DUMP_CFG = _readenv('NUMBA_DUMP_CFG', int, DEBUG_FRONTEND)
    DUMP_IR = _readenv('NUMBA_DUMP_IR', int, DEBUG_FRONTEND)
    DUMP_SSA = _readenv('NUMBA_DUMP_SSA', int, DEBUG_FRONTEND or DEBUG_TYPEINFER)
    DEBUG_ARRAY_OPT = _readenv('NUMBA_DEBUG_ARRAY_OPT', int, 0)
    DEBUG_ARRAY_OPT_RUNTIME = _readenv('NUMBA_DEBUG_ARRAY_OPT_RUNTIME', int, 0)
    DEBUG_ARRAY_OPT_STATS = _readenv('NUMBA_DEBUG_ARRAY_OPT_STATS', int, 0)
    PARALLEL_DIAGNOSTICS = _readenv('NUMBA_PARALLEL_DIAGNOSTICS', int, 0)
    DEBUG_INLINE_CLOSURE = _readenv('NUMBA_DEBUG_INLINE_CLOSURE', int, 0)
    DUMP_LLVM = _readenv('NUMBA_DUMP_LLVM', int, DEBUG)
    DUMP_FUNC_OPT = _readenv('NUMBA_DUMP_FUNC_OPT', int, DEBUG)
    DUMP_OPTIMIZED = _readenv('NUMBA_DUMP_OPTIMIZED', int, DEBUG)
    LOOP_VECTORIZE = _readenv('NUMBA_LOOP_VECTORIZE', int, 1)
    SLP_VECTORIZE = _readenv('NUMBA_SLP_VECTORIZE', int, 0)
    DUMP_ASSEMBLY = _readenv('NUMBA_DUMP_ASSEMBLY', int, DEBUG)
    ANNOTATE = _readenv('NUMBA_DUMP_ANNOTATION', int, 0)
    DIFF_IR = _readenv('NUMBA_DIFF_IR', int, 0)

    def fmt_html_path(path):
        if path is None:
            return path
        else:
            return os.path.abspath(path)
    HTML = _readenv('NUMBA_DUMP_HTML', fmt_html_path, None)

    def avx_default():
        if not _os_supports_avx():
            return False
        else:
            cpu_name = ll.get_host_cpu_name()
            return cpu_name not in ('corei7-avx', 'core-avx-i', 'sandybridge', 'ivybridge')
    ENABLE_AVX = _readenv('NUMBA_ENABLE_AVX', int, avx_default)
    DISABLE_INTEL_SVML = _readenv('NUMBA_DISABLE_INTEL_SVML', int, IS_32BITS)
    DISABLE_JIT = _readenv('NUMBA_DISABLE_JIT', int, 0)
    THREADING_LAYER_PRIORITY = _readenv('NUMBA_THREADING_LAYER_PRIORITY', lambda string: string.split(), ['tbb', 'omp', 'workqueue'])
    THREADING_LAYER = _readenv('NUMBA_THREADING_LAYER', str, 'default')
    CAPTURED_ERRORS = _readenv('NUMBA_CAPTURED_ERRORS', _validate_captured_errors_style, 'old_style')
    CUDA_WARN_ON_IMPLICIT_COPY = _readenv('NUMBA_CUDA_WARN_ON_IMPLICIT_COPY', int, 1)
    FORCE_CUDA_CC = _readenv('NUMBA_FORCE_CUDA_CC', _parse_cc, None)
    CUDA_DEFAULT_PTX_CC = _readenv('NUMBA_CUDA_DEFAULT_PTX_CC', _parse_cc, (5, 0))
    DISABLE_CUDA = _readenv('NUMBA_DISABLE_CUDA', int, int(MACHINE_BITS == 32))
    ENABLE_CUDASIM = _readenv('NUMBA_ENABLE_CUDASIM', int, 0)
    CUDA_LOG_LEVEL = _readenv('NUMBA_CUDA_LOG_LEVEL', str, '')
    CUDA_LOG_API_ARGS = _readenv('NUMBA_CUDA_LOG_API_ARGS', int, 0)
    CUDA_DEALLOCS_COUNT = _readenv('NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT', int, 10)
    CUDA_DEALLOCS_RATIO = _readenv('NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO', float, 0.2)
    CUDA_ARRAY_INTERFACE_SYNC = _readenv('NUMBA_CUDA_ARRAY_INTERFACE_SYNC', int, 1)
    CUDA_DRIVER = _readenv('NUMBA_CUDA_DRIVER', str, '')
    CUDA_LOG_SIZE = _readenv('NUMBA_CUDA_LOG_SIZE', int, 1024)
    CUDA_VERBOSE_JIT_LOG = _readenv('NUMBA_CUDA_VERBOSE_JIT_LOG', int, 1)
    CUDA_PER_THREAD_DEFAULT_STREAM = _readenv('NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM', int, 0)
    CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = _readenv('NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY', int, 0)
    if IS_WIN32:
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            default_cuda_include_path = os.path.join(cuda_path, 'include')
        else:
            default_cuda_include_path = 'cuda_include_not_found'
    else:
        default_cuda_include_path = os.path.join(os.sep, 'usr', 'local', 'cuda', 'include')
    CUDA_INCLUDE_PATH = _readenv('NUMBA_CUDA_INCLUDE_PATH', str, default_cuda_include_path)

    def num_threads_default():
        try:
            sched_getaffinity = os.sched_getaffinity
        except AttributeError:
            pass
        else:
            return max(1, len(sched_getaffinity(0)))
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            return max(1, cpu_count)
        return 1
    NUMBA_DEFAULT_NUM_THREADS = num_threads_default()
    _NUMBA_NUM_THREADS = _readenv('NUMBA_NUM_THREADS', int, NUMBA_DEFAULT_NUM_THREADS)
    if 'NUMBA_NUM_THREADS' in globals() and globals()['NUMBA_NUM_THREADS'] != _NUMBA_NUM_THREADS:
        from numba.np.ufunc import parallel
        if parallel._is_initialized:
            raise RuntimeError('Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched (currently have %s, trying to set %s)' % (_NUMBA_NUM_THREADS, globals()['NUMBA_NUM_THREADS']))
    NUMBA_NUM_THREADS = _NUMBA_NUM_THREADS
    del _NUMBA_NUM_THREADS
    RUNNING_UNDER_PROFILER = 'VS_PROFILER' in os.environ
    ENABLE_PROFILING = _readenv('NUMBA_ENABLE_PROFILING', int, int(RUNNING_UNDER_PROFILER))
    DEBUGINFO_DEFAULT = _readenv('NUMBA_DEBUGINFO', int, ENABLE_PROFILING)
    CUDA_DEBUGINFO_DEFAULT = _readenv('NUMBA_CUDA_DEBUGINFO', int, 0)
    EXTEND_VARIABLE_LIFETIMES = _readenv('NUMBA_EXTEND_VARIABLE_LIFETIMES', int, 0)

    def which_gdb(path_or_bin):
        gdb = shutil.which(path_or_bin)
        return gdb if gdb is not None else path_or_bin
    GDB_BINARY = _readenv('NUMBA_GDB_BINARY', which_gdb, 'gdb')
    CUDA_MEMORY_MANAGER = _readenv('NUMBA_CUDA_MEMORY_MANAGER', str, 'default')
    LLVM_REFPRUNE_PASS = _readenv('NUMBA_LLVM_REFPRUNE_PASS', int, 1)
    LLVM_REFPRUNE_FLAGS = _readenv('NUMBA_LLVM_REFPRUNE_FLAGS', str, 'all' if LLVM_REFPRUNE_PASS else '')
    USE_LLVMLITE_MEMORY_MANAGER = _readenv('NUMBA_USE_LLVMLITE_MEMORY_MANAGER', int, None)
    LLVM_PASS_TIMINGS = _readenv('NUMBA_LLVM_PASS_TIMINGS', int, 0)
    for name, value in locals().copy().items():
        if name.isupper():
            globals()[name] = value