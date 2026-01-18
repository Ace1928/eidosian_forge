import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
def disable_deprecation_warning(self, repo=None):
    """repo is not used yet since _deprecation_warning_done is a global"""
    repository._deprecation_warning_done = True