def add_set(self, event, new_set):
    """
        Add transitions to the states in |new_set| on |event|.
        """
    if type(event) is tuple:
        code0, code1 = event
        i = self.split(code0)
        j = self.split(code1)
        map = self.map
        while i < j:
            map[i + 1].update(new_set)
            i += 2
    else:
        self.get_special(event).update(new_set)