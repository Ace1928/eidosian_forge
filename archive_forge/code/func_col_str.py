from copy import deepcopy
def col_str(column, idx):
    col = '{} {}'.format(self.quote_name(column), self.opclasses[idx])
    try:
        suffix = self.col_suffixes[idx]
        if suffix:
            col = '{} {}'.format(col, suffix)
    except IndexError:
        pass
    return col