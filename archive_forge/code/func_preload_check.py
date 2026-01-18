import ctypes
import os
def preload_check():
    """Raises an exception if the environment is not correctly configured.

  Raises:
    ImportError: If the check detects that the environment is not correctly
      configured, and attempting to load the TensorFlow runtime will fail.
  """
    if os.name == 'nt':
        if MSVCP_DLL_NAMES in build_info.build_info:
            missing = []
            for dll_name in build_info.build_info[MSVCP_DLL_NAMES].split(','):
                try:
                    ctypes.WinDLL(dll_name)
                except OSError:
                    missing.append(dll_name)
            if missing:
                raise ImportError('Could not find the DLL(s) %r. TensorFlow requires that these DLLs be installed in a directory that is named in your %%PATH%% environment variable. You may install these DLLs by downloading "Microsoft C++ Redistributable for Visual Studio 2015, 2017 and 2019" for your platform from this URL: https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads' % ' or '.join(missing))
    else:
        from tensorflow.python.platform import _pywrap_cpu_feature_guard
        _pywrap_cpu_feature_guard.InfoAboutUnusedCPUFeatures()