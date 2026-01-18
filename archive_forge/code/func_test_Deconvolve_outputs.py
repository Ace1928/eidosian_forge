from ..model import Deconvolve
def test_Deconvolve_outputs():
    output_map = dict(cbucket=dict(extensions=None), out_file=dict(extensions=None), reml_script=dict(extensions=None), x1D=dict(extensions=None))
    outputs = Deconvolve.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value