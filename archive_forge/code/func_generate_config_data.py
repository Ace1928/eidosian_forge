from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
def generate_config_data(self, params=None, **kwargs):
    """Load the runtime definition and generate configs from it.

        Args:
            params: (ext_runtime.Params) Runtime parameters.  DEPRECATED.
                Use the keyword args, instead.
            **kwargs: ({str: object, ...}) If specified, these are the
                arguments to the ext_runtime.Params() constructor
                (valid args are at this time are: appinfo, custom and deploy,
                check ext_runtime.Params() for full details)

        Returns:
            ([ext_runtime.GeneratedFile, ...]) Returns list of generated files.

        Raises:
            InvalidRuntime: Couldn't detect a matching runtime.
        """
    configurator = self.maybe_get_configurator(params, **kwargs)
    if not configurator:
        raise InvalidRuntime('Runtime defined in {} did not detect code in {}'.format(self.runtime_def_root, self.temp_path))
    configurator.Prebuild()
    return configurator.GenerateConfigData()