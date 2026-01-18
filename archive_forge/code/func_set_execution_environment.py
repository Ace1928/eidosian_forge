from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import shutil
import tempfile
import unittest
from gae_ext_runtime import ext_runtime
def set_execution_environment(self, exec_env):
    """Set the execution environment used by generate_configs.

        If this is not set, an instance of
        ext_runtime.DefaultExecutionEnvironment is used.

        Args:
            exec_env: (ext_runtime.ExecutionEnvironment) The execution
                environment to be used for config generation.
        """
    self.exec_env = exec_env