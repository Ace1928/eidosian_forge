import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
@classmethod
def output_type_to_ext(cls, outputtype):
    """
        Get the file extension for the given output type.

        Parameters
        ----------
        outputtype : {'NIFTI', 'NIFTI_GZ', 'AFNI'}
            String specifying the output type.

        Returns
        -------
        extension : str
            The file extension for the output type.

        """
    try:
        return cls.ftypes[outputtype]
    except KeyError as e:
        msg = ('Invalid AFNIOUTPUTTYPE: ', outputtype)
        raise KeyError(msg) from e