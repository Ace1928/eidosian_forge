@property
def subsolver_error(self):
    """
        bool : True if subsolver error found for local or global
        separation loop, False otherwise.
        """
    local_subsolver_error = self.solved_locally and self.local_separation_loop_results.subsolver_error
    global_subsolver_error = self.solved_globally and self.global_separation_loop_results.subsolver_error
    return local_subsolver_error or global_subsolver_error