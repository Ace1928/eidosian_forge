import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
def set_R_plotting_device(self, device):
    """
        Set which device R should use to produce plots.
        If device == 'svg' then the package 'Cairo'
        must be installed. Because Cairo forces "onefile=TRUE",
        it is not posible to include multiple plots per cell.

        :param device: ['png', 'jpeg', 'X11', 'svg']
            Device to be used for plotting.
            Currently only 'png', 'jpeg', 'X11' and 'svg' are supported,
            with 'X11' allowing interactive plots on a locally-running jupyter,
            and the other allowing to visualize R figure generated on a remote
            jupyter server/kernel.
        """
    device = device.strip()
    if device not in DEVICES_SUPPORTED:
        raise ValueError(f'device must be one of {DEVICES_SUPPORTED}, got "{device}"')
    if device == 'svg':
        try:
            self.cairo = rpacks.importr('Cairo')
        except ri.embedded.RRuntimeError as rre:
            if rpacks.isinstalled('Cairo'):
                msg = 'An error occurred when trying to load the ' + "R package Cairo'\n%s" % str(rre)
            else:
                msg = textwrap.dedent("\n                    The R package 'Cairo' is required but it does not appear\n                    to be installed/available. Try:\n\n                    import rpy2.robjects.packages as rpacks\n                    utils = rpacks.importr('utils')\n                    utils.chooseCRANmirror(ind=1)\n                    utils.install_packages('Cairo')\n                    ")
            raise RInterpreterError(msg)
    self.device = device