import warnings
import io
from . import utils
import matplotlib
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
@staticmethod
def process_transform(transform, ax=None, data=None, return_trans=False, force_trans=None):
    """Process the transform and convert data to figure or data coordinates

        Parameters
        ----------
        transform : matplotlib Transform object
            The transform applied to the data
        ax : matplotlib Axes object (optional)
            The axes the data is associated with
        data : ndarray (optional)
            The array of data to be transformed.
        return_trans : bool (optional)
            If true, return the final transform of the data
        force_trans : matplotlib.transform instance (optional)
            If supplied, first force the data to this transform

        Returns
        -------
        code : string
            Code is either "data", "axes", "figure", or "display", indicating
            the type of coordinates output.
        transform : matplotlib transform
            the transform used to map input data to output data.
            Returned only if return_trans is True
        new_data : ndarray
            Data transformed to match the given coordinate code.
            Returned only if data is specified
        """
    if isinstance(transform, transforms.BlendedGenericTransform):
        warnings.warn('Blended transforms not yet supported. Zoom behavior may not work as expected.')
    if force_trans is not None:
        if data is not None:
            data = (transform - force_trans).transform(data)
        transform = force_trans
    code = 'display'
    if ax is not None:
        for c, trans in [('data', ax.transData), ('axes', ax.transAxes), ('figure', ax.figure.transFigure), ('display', transforms.IdentityTransform())]:
            if transform.contains_branch(trans):
                code, transform = (c, transform - trans)
                break
    if data is not None:
        if return_trans:
            return (code, transform.transform(data), transform)
        else:
            return (code, transform.transform(data))
    elif return_trans:
        return (code, transform)
    else:
        return code