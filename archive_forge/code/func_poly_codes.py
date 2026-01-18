from matplotlib.path import Path
import numpy as np
import shapely.geometry as sgeom
def poly_codes(poly):
    codes = np.ones(len(poly.xy[0])) * Path.LINETO
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    return codes