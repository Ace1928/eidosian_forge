import copy, logging
from pyomo.common.dependencies import numpy
def solve_tear_direct(self, G, order, function, tears, outEdges, iterLim, tol, tol_type, report_diffs):
    """
        Use direct substitution to solve tears. If multiple tears are
        given they are solved simultaneously.

        Arguments
        ---------
            order
                List of lists of order in which to calculate nodes
            tears
                List of tear edge indexes
            iterLim
                Limit on the number of iterations to run
            tol
                Tolerance at which iteration can be stopped

        Returns
        -------
            list
                List of lists of diff history, differences between input and
                output values at each iteration
        """
    hist = []
    if not len(tears):
        self.run_order(G, order, function, tears)
        return hist
    logger.info('Starting Direct tear convergence')
    ignore = tears + outEdges
    itercount = 0
    while True:
        svals, dvals = self.tear_diff_direct(G, tears)
        err = self.compute_err(svals, dvals, tol_type)
        hist.append(err)
        if report_diffs:
            print('Diff matrix:\n%s' % err)
        if numpy.max(numpy.abs(err)) < tol:
            break
        if itercount >= iterLim:
            logger.warning('Direct failed to converge in %s iterations' % iterLim)
            return hist
        self.pass_tear_direct(G, tears)
        itercount += 1
        logger.info('Running Direct iteration %s' % itercount)
        self.run_order(G, order, function, ignore)
    self.pass_edges(G, outEdges)
    logger.info('Direct converged in %s iterations' % itercount)
    return hist