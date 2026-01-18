def iter_useful(self, struct_iter):
    """Iterate through input tuples, skipping empty ones."""
    for group in struct_iter:
        if len(group[0]) > 0:
            yield group
        elif len(group) > 1 and len(group[1]) > 0:
            yield group