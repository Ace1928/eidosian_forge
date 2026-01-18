from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
def maybe_get_configurator(self, params=None, **kwargs):
    """Load the runtime definition.

        Args:
            params: (ext_runtime.Params) Runtime parameters.  DEPRECATED.
                Use the keyword args, instead.
            **kwargs: ({str: object, ...}) If specified, these are the
                arguments to the ext_runtime.Params() constructor
                (valid args are at this time are: appinfo, custom and deploy,
                check ext_runtime.Params() for full details)

        Returns:
            configurator or None if configurator didn't match
        """
    rt = ext_runtime.ExternalizedRuntime.Load(self.runtime_def_root, self.exec_env)
    params = params or ext_runtime.Params(**kwargs)
    print(params.ToDict())
    configurator = rt.Detect(self.temp_path, params)
    return configurator