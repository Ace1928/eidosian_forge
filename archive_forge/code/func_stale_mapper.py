def stale_mapper(self, encode, value):
    if encode:
        return self.is_stale(value)
    elif value:
        return 0
    else:
        return self.get_flag(0)