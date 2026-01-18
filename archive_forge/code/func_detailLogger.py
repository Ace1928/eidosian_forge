import logging
def detailLogger(self):
    """
        Print information about the iteration to the log.
        """
    logger.info('****** Iteration %d ******' % self.iteration)
    logger.info('trustRadius = %s' % self.trustRadius)
    logger.info('feasibility = %s' % self.feasibility)
    logger.info('objectiveValue = %s' % self.objectiveValue)
    logger.info('stepNorm = %s' % self.stepNorm)
    if self.fStep:
        logger.info('INFO: f-type step')
    if self.thetaStep:
        logger.info('INFO: theta-type step')
    if self.rejected:
        logger.info('INFO: step rejected')