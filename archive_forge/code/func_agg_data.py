from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
def agg_data(self, intent_code=None):
    """
        Aggregate GIFTI data arrays into an ndarray or tuple of ndarray

        In the general case, the numpy data array is extracted from each ``GiftiDataArray``
        object and returned in a ``tuple``, in the order they are found in the GIFTI image.

        If all ``GiftiDataArray`` s have ``intent`` of 2001 (``NIFTI_INTENT_TIME_SERIES``),
        then the data arrays are concatenated as columns, producing a vertex-by-time array.
        If an ``intent_code`` is passed, data arrays are filtered by the selected intents,
        before being aggregated.
        This may be useful for images containing several intents, or ensuring an expected
        data type in an image of uncertain provenance.
        If ``intent_code`` is a ``tuple``, then a ``tuple`` will be returned with the result of
        ``agg_data`` for each element, in order.
        This may be useful for ensuring that expected data arrives in a consistent order.

        Parameters
        ----------
        intent_code : None, string, integer or tuple of strings or integers, optional
            code(s) specifying nifti intent

        Returns
        -------
        tuple of ndarrays or ndarray
            If the input is a tuple, the returned tuple will match the order.

        Examples
        --------

        Consider a surface GIFTI file:

        >>> import nibabel as nib
        >>> from nibabel.testing import get_test_data
        >>> surf_img = nib.load(get_test_data('gifti', 'ascii.gii'))

        The coordinate data, which is indicated by the ``NIFTI_INTENT_POINTSET``
        intent code, may be retrieved using any of the following equivalent
        calls:

        >>> coords = surf_img.agg_data('NIFTI_INTENT_POINTSET')
        >>> coords_2 = surf_img.agg_data('pointset')
        >>> coords_3 = surf_img.agg_data(1008)  # Numeric code for pointset
        >>> print(np.array2string(coords, precision=3))
        [[-16.072 -66.188  21.267]
         [-16.706 -66.054  21.233]
         [-17.614 -65.402  21.071]]
        >>> np.array_equal(coords, coords_2)
        True
        >>> np.array_equal(coords, coords_3)
        True

        Similarly, the triangle mesh can be retrieved using various intent
        specifiers:

        >>> triangles = surf_img.agg_data('NIFTI_INTENT_TRIANGLE')
        >>> triangles_2 = surf_img.agg_data('triangle')
        >>> triangles_3 = surf_img.agg_data(1009)  # Numeric code for pointset
        >>> print(np.array2string(triangles))
        [[0 1 2]]
        >>> np.array_equal(triangles, triangles_2)
        True
        >>> np.array_equal(triangles, triangles_3)
        True

        All arrays can be retrieved as a ``tuple`` by omitting the intent
        code:

        >>> coords_4, triangles_4 = surf_img.agg_data()
        >>> np.array_equal(coords, coords_4)
        True
        >>> np.array_equal(triangles, triangles_4)
        True

        Finally, a tuple of intent codes may be passed in order to select
        the arrays in a specific order:

        >>> triangles_5, coords_5 = surf_img.agg_data(('triangle', 'pointset'))
        >>> np.array_equal(triangles, triangles_5)
        True
        >>> np.array_equal(coords, coords_5)
        True

        The following image is a GIFTI file with ten (10) data arrays of the same
        size, and with intent code 2001 (``NIFTI_INTENT_TIME_SERIES``):

        >>> func_img = nib.load(get_test_data('gifti', 'task.func.gii'))

        When aggregating time series data, these arrays are concatenated into
        a single, vertex-by-timestep array:

        >>> series = func_img.agg_data()
        >>> series.shape
        (642, 10)

        In the case of a GIFTI file with unknown data arrays, it may be preferable
        to specify the intent code, so that a time series array is always returned:

        >>> series_2 = func_img.agg_data('NIFTI_INTENT_TIME_SERIES')
        >>> series_3 = func_img.agg_data('time series')
        >>> series_4 = func_img.agg_data(2001)
        >>> np.array_equal(series, series_2)
        True
        >>> np.array_equal(series, series_3)
        True
        >>> np.array_equal(series, series_4)
        True

        Requesting a data array from a GIFTI file with no matching intent codes
        will result in an empty tuple:

        >>> surf_img.agg_data('time series')
        ()
        >>> func_img.agg_data('triangle')
        ()
        """
    if isinstance(intent_code, tuple):
        return tuple((self.agg_data(intent_code=code) for code in intent_code))
    darrays = self.darrays if intent_code is None else self.get_arrays_from_intent(intent_code)
    all_data = tuple((da.data for da in darrays))
    all_intent = {intent_codes.niistring[da.intent] for da in darrays}
    if all_intent == {'NIFTI_INTENT_TIME_SERIES'}:
        return np.column_stack(all_data)
    if len(all_data) == 1:
        all_data = all_data[0]
    return all_data