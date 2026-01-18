import os.path
import sys
from _pydevd_bundle.pydevd_constants import Null
def start_coverage_support(configuration):
    return start_coverage_support_from_params(configuration.coverage_output_dir, configuration.coverage_output_file, configuration.jobs, configuration.coverage_include)