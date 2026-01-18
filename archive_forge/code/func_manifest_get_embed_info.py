import os
import subprocess
import sys
import re
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
import winreg
def manifest_get_embed_info(self, target_desc, ld_args):
    for arg in ld_args:
        if arg.startswith('/MANIFESTFILE:'):
            temp_manifest = arg.split(':', 1)[1]
            break
    else:
        return None
    if target_desc == CCompiler.EXECUTABLE:
        mfid = 1
    else:
        mfid = 2
        temp_manifest = self._remove_visual_c_ref(temp_manifest)
    if temp_manifest is None:
        return None
    return (temp_manifest, mfid)