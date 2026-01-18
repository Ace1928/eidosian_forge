import warnings
import io
from . import utils
import matplotlib
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
def draw_collection(self, ax, collection, force_pathtrans=None, force_offsettrans=None):
    """Process a matplotlib collection and call renderer.draw_collection"""
    transform, transOffset, offsets, paths = collection._prepare_points()
    offset_coords, offsets = self.process_transform(transOffset, ax, offsets, force_trans=force_offsettrans)
    path_coords = self.process_transform(transform, ax, force_trans=force_pathtrans)
    processed_paths = [utils.SVG_path(path) for path in paths]
    processed_paths = [(self.process_transform(transform, ax, path[0], force_trans=force_pathtrans)[1], path[1]) for path in processed_paths]
    path_transforms = collection.get_transforms()
    try:
        path_transforms = [t.get_matrix() for t in path_transforms]
    except AttributeError:
        pass
    styles = {'linewidth': collection.get_linewidths(), 'facecolor': collection.get_facecolors(), 'edgecolor': collection.get_edgecolors(), 'alpha': collection._alpha, 'zorder': collection.get_zorder()}
    offset_dict = {'data': 'before', 'screen': 'after'}
    offset_order = offset_dict[collection.get_offset_position()] if hasattr(collection, 'get_offset_position') else 'after'
    self.renderer.draw_path_collection(paths=processed_paths, path_coordinates=path_coords, path_transforms=path_transforms, offsets=offsets, offset_coordinates=offset_coords, offset_order=offset_order, styles=styles, mplobj=collection)