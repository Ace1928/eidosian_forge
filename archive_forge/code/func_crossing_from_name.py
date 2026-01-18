import spherogram
def crossing_from_name(link, crossname):
    for c in link.crossings:
        if c.label == crossname:
            return c
    raise ValueError('crossing not found')