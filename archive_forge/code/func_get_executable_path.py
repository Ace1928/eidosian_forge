import os
from absl import flags
def get_executable_path(py_binary_name):
    """Returns the executable path of a py_binary.

  This returns the executable path of a py_binary that is in another Bazel
  target's data dependencies.

  On Linux/macOS, the path and __file__ has the same root directory.
  On Windows, bazel builds an .exe file and we need to use the MANIFEST file
  the location the actual binary.

  Args:
    py_binary_name: string, the name of a py_binary that is in another Bazel
        target's data dependencies.

  Raises:
    RuntimeError: Raised when it cannot locate the executable path.
  """
    if os.name == 'nt':
        py_binary_name += '.exe'
        manifest_file = os.path.join(FLAGS.test_srcdir, 'MANIFEST')
        workspace_name = os.environ['TEST_WORKSPACE']
        manifest_entry = '{}/{}'.format(workspace_name, py_binary_name)
        with open(manifest_file, 'r') as manifest_fd:
            for line in manifest_fd:
                tokens = line.strip().split(' ')
                if len(tokens) != 2:
                    continue
                if manifest_entry == tokens[0]:
                    return tokens[1]
        raise RuntimeError('Cannot locate executable path for {}, MANIFEST file: {}.'.format(py_binary_name, manifest_file))
    else:
        path = __file__
        for _ in range(__name__.count('.') + 1):
            path = os.path.dirname(path)
        root_directory = path
        return os.path.join(root_directory, py_binary_name)