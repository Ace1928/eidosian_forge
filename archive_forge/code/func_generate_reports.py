import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def generate_reports(self, test_runner):
    """Generates the XML reports to a given XMLTestRunner object."""
    all_results = self._get_info_by_testcase()
    if isinstance(test_runner.output, str) and (not os.path.exists(test_runner.output)):
        os.makedirs(test_runner.output)
    for suite, tests in all_results.items():
        doc = XMLDocument()
        testsuite = _XMLTestResult._report_testsuite(suite, tests, doc)
        stdout, stderr = ([], [])
        for test in tests:
            _XMLTestResult._report_testcase(suite, test, testsuite, doc)
            if test.stdout:
                stdout.extend(['*****************', test.get_description(), test.stdout])
            if test.stderr:
                stderr.extend(['*****************', test.get_description(), test.stderr])
        _XMLTestResult._report_output(test_runner, testsuite, doc, '\n'.join(stdout), '\n'.join(stderr))
        xml_content = doc.toprettyxml(indent='\t')
        if type(test_runner.output) is str:
            report_file = open('%s%sTEST-%s.xml' % (test_runner.output, os.sep, suite), 'w')
            try:
                report_file.write(xml_content)
            finally:
                report_file.close()
        else:
            test_runner.output.write(xml_content)