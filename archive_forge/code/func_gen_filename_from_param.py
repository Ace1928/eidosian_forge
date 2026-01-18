import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def gen_filename_from_param(param, base):
    fileExtensions = param.getAttribute('fileExtensions')
    if fileExtensions:
        firstFileExtension = fileExtensions.split(',')[0]
        ext = firstFileExtension
    else:
        ext = {'image': '.nii', 'transform': '.mat', 'file': '', 'directory': '', 'geometry': '.vtk'}[param.nodeName]
    return base + ext