import numpy
import numpy as np
from . import _ni_support
from . import _ni_label
from . import _nd_image
from . import _morphology
def find_objects(input, max_label=0):
    """
    Find objects in a labeled array.

    Parameters
    ----------
    input : ndarray of ints
        Array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        given, the positions of all objects are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice. The label ``l`` corresponds to
        the index ``l-1`` in the returned list.

    See Also
    --------
    label, center_of_mass

    Notes
    -----
    This function is very useful for isolating a volume of interest inside
    a 3-D array, that cannot be "seen through".

    Examples
    --------
    >>> from scipy import ndimage
    >>> import numpy as np
    >>> a = np.zeros((6,6), dtype=int)
    >>> a[2:4, 2:4] = 1
    >>> a[4, 4] = 1
    >>> a[:2, :3] = 2
    >>> a[0, 5] = 3
    >>> a
    array([[2, 2, 2, 0, 0, 3],
           [2, 2, 2, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> ndimage.find_objects(a)
    [(slice(2, 5, None), slice(2, 5, None)),
     (slice(0, 2, None), slice(0, 3, None)),
     (slice(0, 1, None), slice(5, 6, None))]
    >>> ndimage.find_objects(a, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
    >>> ndimage.find_objects(a == 1, max_label=2)
    [(slice(2, 5, None), slice(2, 5, None)), None]

    >>> loc = ndimage.find_objects(a)[0]
    >>> a[loc]
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 1]])

    """
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if max_label < 1:
        max_label = input.max()
    return _nd_image.find_objects(input, max_label)