from glob import glob
import os
from ... import logging
from ...utils.filemanip import fname_presuffix
from ..base import traits, isdefined, CommandLine, CommandLineInputSpec, PackageInfo
from ...external.due import BibTeX
def check_fsl():
    ver = Info.version()
    if ver:
        return 0
    else:
        return 1