import os
import platform
import logging
def find_conf_path(prefix='tvmop'):
    """Find TVM op config files.

    Returns
    -------
    conf_path : string
        Path to the config files.
    """
    conf_from_env = os.environ.get('MXNET_CONF_PATH')
    if conf_from_env:
        if os.path.isfile(conf_from_env):
            if not os.path.isabs(conf_from_env):
                logging.warning('MXNET_CONF_PATH should be an absolute path, instead of: %s', conf_from_env)
            else:
                return conf_from_env
        else:
            logging.warning("MXNET_CONF_PATH '%s' doesn't exist", conf_from_env)
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    makefile_path = os.path.join(curr_path, '../../lib/')
    cmake_build_path = os.path.join(curr_path, '../../build/')
    candidates_path = [makefile_path, cmake_build_path]
    candidates_path = [p + prefix + '.conf' for p in candidates_path]
    conf_path = [p for p in candidates_path if os.path.exists(p) and os.path.isfile(p)]
    if len(conf_path) == 0:
        raise RuntimeError('Cannot find the TVM op config.\n' + 'List of candidates:\n' + str('\n'.join(candidates_path)))
    return conf_path