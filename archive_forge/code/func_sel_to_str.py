from typing import Union
def sel_to_str(self, sel: dict, isel: dict):
    """WIP."""
    if sel:
        return ', '.join([self.dim_coord_to_str(dim, v, i) for (dim, v), (_, i) in zip(sel.items(), isel.items())])
    return ''