import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
@classmethod
def subjectsdir(cls):
    """Check the global SUBJECTS_DIR

        Parameters
        ----------

        subjects_dir :  string
            The system defined subjects directory

        Returns
        -------

        subject_dir : string
            Represents the current environment setting of SUBJECTS_DIR

        """
    if cls.version():
        return os.environ['SUBJECTS_DIR']
    return None