import os
import platform
import logging
def find_include_path():
    """Find MXNet included header files.

    Returns
    -------
    incl_path : string
        Path to the header files.
    """
    incl_from_env = os.environ.get('MXNET_INCLUDE_PATH')
    if incl_from_env:
        if os.path.isdir(incl_from_env):
            if not os.path.isabs(incl_from_env):
                logging.warning('MXNET_INCLUDE_PATH should be an absolute path, instead of: %s', incl_from_env)
            else:
                return incl_from_env
        else:
            logging.warning("MXNET_INCLUDE_PATH '%s' doesn't exist", incl_from_env)
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    pip_incl_path = os.path.join(curr_path, 'include/')
    if os.path.isdir(pip_incl_path):
        return pip_incl_path
    else:
        src_incl_path = os.path.join(curr_path, '../../include/')
        if os.path.isdir(src_incl_path):
            return src_incl_path
        else:
            raise RuntimeError('Cannot find the MXNet include path in either ' + pip_incl_path + ' or ' + src_incl_path + '\n')