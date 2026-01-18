import os
def server_binary():
    """Return the path to a TensorBoard data server binary, if possible.

    Returns:
      A string path on disk, or `None` if there is no binary bundled
      with this package.
    """
    path = os.path.join(os.path.dirname(__file__), 'bin', 'server')
    if not os.path.exists(path):
        return None
    return path