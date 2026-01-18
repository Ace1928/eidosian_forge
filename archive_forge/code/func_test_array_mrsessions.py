import unittest
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import logging as log
@skip_if_no_network
def test_array_mrsessions(self):
    """
        From a given subject which has multiple types of experiments, get a
        list of MRI sessions and assert its length matches the list of
        experiments of type 'xnat:mrSessionData'
        """
    mris = self._intf.array.mrsessions(subject_id='XNAT_S04207').data
    e = self._intf.array.experiments(subject_id='XNAT_S04207', experiment_type='xnat:mrSessionData')
    exps = e.data
    self.assertListEqual(mris, exps)