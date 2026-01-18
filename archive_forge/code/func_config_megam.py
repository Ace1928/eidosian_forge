import subprocess
from nltk.internals import find_binary
def config_megam(bin=None):
    """
    Configure NLTK's interface to the ``megam`` maxent optimization
    package.

    :param bin: The full path to the ``megam`` binary.  If not specified,
        then nltk will search the system for a ``megam`` binary; and if
        one is not found, it will raise a ``LookupError`` exception.
    :type bin: str
    """
    global _megam_bin
    _megam_bin = find_binary('megam', bin, env_vars=['MEGAM'], binary_names=['megam.opt', 'megam', 'megam_686', 'megam_i686.opt'], url='https://www.umiacs.umd.edu/~hal/megam/index.html')