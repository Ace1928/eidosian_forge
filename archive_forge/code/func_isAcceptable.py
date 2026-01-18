def isAcceptable(self, filterElement, maximum_feasibility):
    """
        Check whether a step is acceptable to the filter.
        If not, we reject the step.
        """
    if filterElement.feasible > maximum_feasibility:
        return False
    for fe in self.TrustRegionFilter:
        if fe.compare(filterElement) == -1:
            return False
    return True