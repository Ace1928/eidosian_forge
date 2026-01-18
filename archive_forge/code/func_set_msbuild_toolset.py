import hashlib
import os
import random
from operator import attrgetter
import gyp.common
def set_msbuild_toolset(self, msbuild_toolset):
    self.msbuild_toolset = msbuild_toolset