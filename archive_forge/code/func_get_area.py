import numpy as np
def get_area(y, x):
    """Get the area under the curve."""
    return trapz(y=y, x=x)