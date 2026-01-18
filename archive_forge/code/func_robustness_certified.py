@property
def robustness_certified(self):
    """
        bool : Return True if separation results certify that
        first-stage solution is robust, False otherwise.
        """
    assert self.solved_locally or self.solved_globally
    if self.time_out or self.subsolver_error:
        return False
    if self.solved_locally:
        heuristically_robust = not self.local_separation_loop_results.found_violation
    else:
        heuristically_robust = None
    if self.solved_globally:
        is_robust = not self.global_separation_loop_results.found_violation
    else:
        is_robust = heuristically_robust
    return is_robust