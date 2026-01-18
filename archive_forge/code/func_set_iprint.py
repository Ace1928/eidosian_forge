import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
def set_iprint(self, init=None, so_init=None, iter=None, so_iter=None, iter_step=None, final=None, so_final=None):
    """ Set the iprint parameter for the printing of computation reports.

        If any of the arguments are specified here, then they are set in the
        iprint member. If iprint is not set manually or with this method, then
        ODRPACK defaults to no printing. If no filename is specified with the
        member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to
        print to stdout in addition to the specified filename by setting the
        so_* arguments to this function, but one cannot specify to print to
        stdout but not a file since one can do that by not specifying a rptfile
        filename.

        There are three reports: initialization, iteration, and final reports.
        They are represented by the arguments init, iter, and final
        respectively.  The permissible values are 0, 1, and 2 representing "no
        report", "short report", and "long report" respectively.

        The argument iter_step (0 <= iter_step <= 9) specifies how often to make
        the iteration report; the report will be made for every iter_step'th
        iteration starting with iteration one. If iter_step == 0, then no
        iteration report is made, regardless of the other arguments.

        If the rptfile is None, then any so_* arguments supplied will raise an
        exception.
        """
    if self.iprint is None:
        self.iprint = 0
    ip = [self.iprint // 1000 % 10, self.iprint // 100 % 10, self.iprint // 10 % 10, self.iprint % 10]
    ip2arg = [[0, 0], [1, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]
    if self.rptfile is None and (so_init is not None or so_iter is not None or so_final is not None):
        raise OdrError('no rptfile specified, cannot output to stdout twice')
    iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
    if init is not None:
        iprint_l[0] = init
    if so_init is not None:
        iprint_l[1] = so_init
    if iter is not None:
        iprint_l[2] = iter
    if so_iter is not None:
        iprint_l[3] = so_iter
    if final is not None:
        iprint_l[4] = final
    if so_final is not None:
        iprint_l[5] = so_final
    if iter_step in range(10):
        ip[2] = iter_step
    ip[0] = ip2arg.index(iprint_l[0:2])
    ip[1] = ip2arg.index(iprint_l[2:4])
    ip[3] = ip2arg.index(iprint_l[4:6])
    self.iprint = ip[0] * 1000 + ip[1] * 100 + ip[2] * 10 + ip[3]