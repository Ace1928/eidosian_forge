import json
import locale
import multiprocessing
import os
import platform
import textwrap
import sys
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from subprocess import check_output, PIPE, CalledProcessError
import numpy as np
import llvmlite.binding as llvmbind
from llvmlite import __version__ as llvmlite_version
from numba import cuda as cu, __version__ as version_number
from numba.cuda import cudadrv
from numba.cuda.cudadrv.driver import driver as cudriver
from numba.cuda.cudadrv.runtime import runtime as curuntime
from numba.core import config
def get_sysinfo():
    sys_info = {_start: datetime.now(), _start_utc: datetime.utcnow(), _machine: platform.machine(), _cpu_name: llvmbind.get_host_cpu_name(), _cpu_count: multiprocessing.cpu_count(), _platform_name: platform.platform(aliased=True), _platform_release: platform.release(), _os_name: platform.system(), _os_version: platform.version(), _python_comp: platform.python_compiler(), _python_impl: platform.python_implementation(), _python_version: platform.python_version(), _numba_env_vars: {k: v for k, v in os.environ.items() if k.startswith('NUMBA_')}, _numba_version: version_number, _llvm_version: '.'.join((str(i) for i in llvmbind.llvm_version_info)), _llvmlite_version: llvmlite_version, _psutil: _psutil_import}
    try:
        feature_map = llvmbind.get_host_cpu_features()
    except RuntimeError as e:
        _error_log.append(f'Error (CPU features): {e}')
    else:
        features = sorted([key for key, value in feature_map.items() if value])
        sys_info[_cpu_features] = ' '.join(features)
    try:
        sys_info[_python_locale] = '.'.join([str(i) for i in locale.getdefaultlocale()])
    except Exception as e:
        _error_log.append(f'Error (locale): {e}')
    try:
        cu.list_devices()[0]
    except Exception as e:
        sys_info[_cu_dev_init] = False
        msg_not_found = 'CUDA driver library cannot be found'
        msg_disabled_by_user = 'CUDA is disabled'
        msg_end = ' or no CUDA enabled devices are present.'
        msg_generic_problem = 'CUDA device initialisation problem.'
        msg = getattr(e, 'msg', None)
        if msg is not None:
            if msg_not_found in msg:
                err_msg = msg_not_found + msg_end
            elif msg_disabled_by_user in msg:
                err_msg = msg_disabled_by_user + msg_end
            else:
                err_msg = msg_generic_problem + ' Message:' + msg
        else:
            err_msg = msg_generic_problem + ' ' + str(e)
        _warning_log.append('Warning (cuda): %s\nException class: %s' % (err_msg, str(type(e))))
    else:
        try:
            sys_info[_cu_dev_init] = True
            output = StringIO()
            with redirect_stdout(output):
                cu.detect()
            sys_info[_cu_detect_out] = output.getvalue()
            output.close()
            cu_drv_ver = cudriver.get_version()
            cu_rt_ver = curuntime.get_version()
            sys_info[_cu_drv_ver] = '%s.%s' % cu_drv_ver
            sys_info[_cu_rt_ver] = '%s.%s' % cu_rt_ver
            output = StringIO()
            with redirect_stdout(output):
                cudadrv.libs.test()
            sys_info[_cu_lib_test] = output.getvalue()
            output.close()
            try:
                from cuda import cuda
                nvidia_bindings_available = True
            except ImportError:
                nvidia_bindings_available = False
            sys_info[_cu_nvidia_bindings] = nvidia_bindings_available
            nv_binding_used = bool(cudadrv.driver.USE_NV_BINDING)
            sys_info[_cu_nvidia_bindings_used] = nv_binding_used
            try:
                from ptxcompiler import compile_ptx
                from cubinlinker import CubinLinker
                sys_info[_cu_mvc_available] = True
            except ImportError:
                sys_info[_cu_mvc_available] = False
            sys_info[_cu_mvc_needed] = cu_rt_ver > cu_drv_ver
            sys_info[_cu_mvc_in_use] = bool(config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY)
        except Exception as e:
            _warning_log.append(f'Warning (cuda): Probing CUDA failed (device and driver present, runtime problem?)\n(cuda) {type(e)}: {e}')
    sys_info[_numpy_version] = np.version.full_version
    try:
        from numpy.core._multiarray_umath import __cpu_features__, __cpu_dispatch__, __cpu_baseline__
    except ImportError:
        sys_info[_numpy_AVX512_SKX_detected] = False
    else:
        feat_filtered = [k for k, v in __cpu_features__.items() if v]
        sys_info[_numpy_supported_simd_features] = feat_filtered
        sys_info[_numpy_supported_simd_dispatch] = __cpu_dispatch__
        sys_info[_numpy_supported_simd_baseline] = __cpu_baseline__
        sys_info[_numpy_AVX512_SKX_detected] = __cpu_features__.get('AVX512_SKX', False)
    svml_lib_loaded = True
    try:
        if sys.platform.startswith('linux'):
            llvmbind.load_library_permanently('libsvml.so')
        elif sys.platform.startswith('darwin'):
            llvmbind.load_library_permanently('libsvml.dylib')
        elif sys.platform.startswith('win'):
            llvmbind.load_library_permanently('svml_dispmd')
        else:
            svml_lib_loaded = False
    except Exception:
        svml_lib_loaded = False
    func = getattr(llvmbind.targets, 'has_svml', None)
    sys_info[_llvm_svml_patched] = func() if func else False
    sys_info[_svml_state] = config.USING_SVML
    sys_info[_svml_loaded] = svml_lib_loaded
    sys_info[_svml_operational] = all((sys_info[_svml_state], sys_info[_svml_loaded], sys_info[_llvm_svml_patched]))

    def parse_error(e, backend):
        try:
            path, problem, symbol = [x.strip() for x in e.msg.split(':')]
            extn_dso = os.path.split(path)[1]
            if backend in extn_dso:
                return '%s: %s' % (problem, symbol)
        except Exception:
            pass
        return 'Unknown import problem.'
    try:
        from numba.np.ufunc import tbbpool
        from numba.np.ufunc.parallel import _check_tbb_version_compatible
        _check_tbb_version_compatible()
        sys_info[_tbb_thread] = True
    except ImportError as e:
        sys_info[_tbb_thread] = False
        sys_info[_tbb_error] = parse_error(e, 'tbbpool')
    try:
        from numba.np.ufunc import omppool
        sys_info[_openmp_thread] = True
        sys_info[_openmp_vendor] = omppool.openmp_vendor
    except ImportError as e:
        sys_info[_openmp_thread] = False
        sys_info[_openmp_error] = parse_error(e, 'omppool')
    try:
        from numba.np.ufunc import workqueue
        sys_info[_wkq_thread] = True
    except ImportError as e:
        sys_info[_wkq_thread] = True
        sys_info[_wkq_error] = parse_error(e, 'workqueue')
    cmd = ('conda', 'info', '--json')
    try:
        conda_out = check_output(cmd)
    except Exception as e:
        _warning_log.append(f'Warning: Conda not available.\n Error was {e}\n')
        cmd = (sys.executable, '-m', 'pip', 'list')
        try:
            reqs = check_output(cmd)
        except Exception as e:
            _error_log.append(f'Error (pip): {e}')
        else:
            sys_info[_inst_pkg] = reqs.decode().splitlines()
    else:
        jsond = json.loads(conda_out.decode())
        keys = {'conda_build_version': _conda_build_ver, 'conda_env_version': _conda_env_ver, 'platform': _conda_platform, 'python_version': _conda_python_ver, 'root_writable': _conda_root_writable}
        for conda_k, sysinfo_k in keys.items():
            sys_info[sysinfo_k] = jsond.get(conda_k, 'N/A')
        cmd = ('conda', 'list')
        try:
            conda_out = check_output(cmd)
        except CalledProcessError as e:
            _error_log.append(f'Error (conda): {e}')
        else:
            data = conda_out.decode().splitlines()
            sys_info[_inst_pkg] = [l for l in data if not l.startswith('#')]
    sys_info.update(get_os_spec_info(sys_info[_os_name]))
    sys_info[_errors] = _error_log
    sys_info[_warnings] = _warning_log
    sys_info[_runtime] = (datetime.now() - sys_info[_start]).total_seconds()
    return sys_info