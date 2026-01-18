@property
def solved_globally(self):
    """
        bool : True if global separation loop was invoked,
        False otherwise.
        """
    return self.global_separation_loop_results is not None