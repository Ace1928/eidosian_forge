from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_inputLabelRow(self):
    """
        The table returned by L{tableMaker} always contains the input
        symbol label in its first row, and that row contains one cell
        with a port attribute set to the provided port.
        """

    def hasPort(element):
        return not isLeaf(element) and element.attributes.get('port') == self.port
    for outputLabels in ([], ['an output label']):
        table = self.tableMaker(self.inputLabel, outputLabels, port=self.port)
        self.assertGreater(len(table.children), 0)
        inputLabelRow = table.children[0]
        portCandidates = findElements(table, hasPort)
        self.assertEqual(len(portCandidates), 1)
        self.assertEqual(portCandidates[0].name, 'td')
        self.assertEqual(findElements(inputLabelRow, isLeaf), [self.inputLabel])