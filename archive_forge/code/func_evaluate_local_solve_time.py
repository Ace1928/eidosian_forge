def evaluate_local_solve_time(self, evaluator_func, **evaluator_func_kwargs):
    """
        Evaluate total time required by local subordinate solvers
        for separation problem of interest.

        Parameters
        ----------
        evaluator_func : callable
            Solve time evaluator function.
            This callable should accept an object of type
            ``pyomo.opt.results.SolverResults``, and
            return a float equal to the time required.
        **evaluator_func_kwargs : dict, optional
            Keyword arguments to evaluator function.

        Returns
        -------
        float
            Total time spent by local solvers.
        """
    if self.solved_locally:
        return self.local_separation_loop_results.evaluate_total_solve_time(evaluator_func, **evaluator_func_kwargs)
    else:
        return 0