from __future__ import annotations
import os
from ...io import (
from ...util import (
from ...util_common import (
from ...provisioning import (
from .combine import (
from . import (
def command_coverage_html(args: CoverageHtmlConfig) -> None:
    """Generate an HTML coverage report."""
    host_state = prepare_profiles(args)
    output_files = combine_coverage_files(args, host_state)
    for output_file in output_files:
        if output_file.endswith('-powershell'):
            display.info('Skipping output file %s in html generation' % output_file, verbosity=3)
            continue
        dir_name = os.path.join(ResultType.REPORTS.path, os.path.basename(output_file))
        make_dirs(dir_name)
        run_coverage(args, host_state, output_file, 'html', ['-i', '-d', dir_name])
        display.info('HTML report generated: file:///%s' % os.path.join(dir_name, 'index.html'))