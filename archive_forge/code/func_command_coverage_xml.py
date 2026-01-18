from __future__ import annotations
import os
import time
from xml.etree.ElementTree import (
from xml.dom import (
from ...io import (
from ...util_common import (
from ...util import (
from ...data import (
from ...provisioning import (
from .combine import (
from . import (
def command_coverage_xml(args: CoverageXmlConfig) -> None:
    """Generate an XML coverage report."""
    host_state = prepare_profiles(args)
    output_files = combine_coverage_files(args, host_state)
    for output_file in output_files:
        xml_name = '%s.xml' % os.path.basename(output_file)
        if output_file.endswith('-powershell'):
            report = _generate_powershell_xml(output_file)
            rough_string = tostring(report, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty = reparsed.toprettyxml(indent='    ')
            write_text_test_results(ResultType.REPORTS, xml_name, pretty)
        else:
            xml_path = os.path.join(ResultType.REPORTS.path, xml_name)
            make_dirs(ResultType.REPORTS.path)
            run_coverage(args, host_state, output_file, 'xml', ['-i', '-o', xml_path])