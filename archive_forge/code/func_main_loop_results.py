@property
def main_loop_results(self):
    """
        SeparationLoopResults : Main separation loop results.
        In particular, this is considered to be the global
        loop result if solved globally, and the local loop
        results otherwise.
        """
    if self.solved_globally:
        return self.global_separation_loop_results
    return self.local_separation_loop_results