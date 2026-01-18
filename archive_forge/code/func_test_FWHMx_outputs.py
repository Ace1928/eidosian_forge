from ..utils import FWHMx
def test_FWHMx_outputs():
    output_map = dict(acf_param=dict(), fwhm=dict(), out_acf=dict(extensions=None), out_detrend=dict(extensions=None), out_file=dict(extensions=None), out_subbricks=dict(extensions=None))
    outputs = FWHMx.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value