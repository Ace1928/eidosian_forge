def termination_acceptable(self, acceptable_terminations):
    """
        Return True if termination condition for at least
        one result in `self.results_list` is in list
        of pre-specified acceptable terminations, False otherwise.

        Parameters
        ----------
        acceptable_terminations : set of pyomo.opt.TerminationCondition
            Acceptable termination conditions.

        Returns
        -------
        bool
        """
    return any((res.solver.termination_condition in acceptable_terminations for res in self.results_list))