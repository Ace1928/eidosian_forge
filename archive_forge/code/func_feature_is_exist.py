import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_is_exist(self, name):
    """
        Returns True if a certain feature is exist and covered within
        ``_Config.conf_features``.

        Parameters
        ----------
        'name': str
            feature name in uppercase.
        """
    assert name.isupper()
    return name in self.conf_features