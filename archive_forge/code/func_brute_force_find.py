def brute_force_find(self, interval):
    """
        Search plain list as ground truth to compare against.
        """
    return [entry[1] for entry in self._entries if entry[0].overlaps(interval)]