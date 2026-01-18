import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
@classmethod
def from_image(klass, img):
    """Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : instance
            In fact, an object with the API of :class:`DataobjImage`.

        Returns
        -------
        cimg : instance
            Image, of our own class
        """
    if isinstance(img, klass):
        return img
    raise NotImplementedError