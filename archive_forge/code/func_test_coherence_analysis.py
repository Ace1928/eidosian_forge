import os
import tempfile
import numpy as np
import pytest
from nipype.testing import example_data
import nipype.interfaces.nitime as nitime
@pytest.mark.skipif(no_nitime, reason='nitime is not installed')
def test_coherence_analysis(tmpdir):
    """Test that the coherence analyzer works"""
    import nitime.analysis as nta
    import nitime.timeseries as ts
    tmpdir.chdir()
    CA = nitime.CoherenceAnalyzer()
    CA.inputs.TR = 1.89
    CA.inputs.in_file = example_data('fmri_timeseries.csv')
    if display_available:
        tmp_png = tempfile.mkstemp(suffix='.png')[1]
        CA.inputs.output_figure_file = tmp_png
    tmp_csv = tempfile.mkstemp(suffix='.csv')[1]
    CA.inputs.output_csv_file = tmp_csv
    o = CA.run()
    assert o.outputs.coherence_array.shape == (31, 31)
    TR = 1.89
    data_rec = np.recfromcsv(example_data('fmri_timeseries.csv'))
    roi_names = np.array(data_rec.dtype.names)
    n_samples = data_rec.shape[0]
    data = np.zeros((len(roi_names), n_samples))
    for n_idx, roi in enumerate(roi_names):
        data[n_idx] = data_rec[roi]
    T = ts.TimeSeries(data, sampling_interval=TR)
    assert (CA._csv2ts().data == T.data).all()
    T.metadata['roi'] = roi_names
    C = nta.CoherenceAnalyzer(T, method=dict(this_method='welch', NFFT=CA.inputs.NFFT, n_overlap=CA.inputs.n_overlap))
    freq_idx = np.where((C.frequencies > CA.inputs.frequency_range[0]) * (C.frequencies < CA.inputs.frequency_range[1]))[0]
    coh = np.mean(C.coherence[:, :, freq_idx], -1)
    assert (o.outputs.coherence_array == coh).all()