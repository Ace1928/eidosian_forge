@property
def scaled_violations(self):
    """
        None or ComponentMap : Scaled performance constraint violations
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_perf_con``
        set to None for both local and global loop results),
        then None is returned.
        """
    return self.get_violating_attr('scaled_violations')