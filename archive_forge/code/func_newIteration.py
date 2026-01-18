import logging
def newIteration(self, iteration, feasibility, objectiveValue, trustRadius, stepNorm):
    """
        Add a new iteration to the list of iterations
        """
    self.iterrecord = IterationRecord(iteration, feasibility=feasibility, objectiveValue=objectiveValue, trustRadius=trustRadius, stepNorm=stepNorm)
    self.iterations.append(self.iterrecord)