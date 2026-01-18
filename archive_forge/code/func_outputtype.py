import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
@classmethod
def outputtype(cls):
    """
        Set default output filetype.

        AFNI has no environment variables, Output filetypes get set in command line calls
        Nipype uses ``AFNI`` as default


        Returns
        -------
        None

        """
    return 'AFNI'