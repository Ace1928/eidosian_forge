def get_special(self, event):
    """
        Get state set for special event, adding a new entry if necessary.
        """
    special = self.special
    set = special.get(event, None)
    if not set:
        set = {}
        special[event] = set
    return set