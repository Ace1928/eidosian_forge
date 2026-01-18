import unittest
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import logging as log
@skip_if_no_network
def test_array_scans(self):
    """
        Get a list of scans from a given experiment which has multiple types
        of scans (i.e. PETScans and CTScans) and assert it gathers them all.
        """
    s = self._intf.array.scans(experiment_id='XNAT_E16718').data
    self.assertEqual(len(s), 1)