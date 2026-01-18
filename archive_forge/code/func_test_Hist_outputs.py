from ..preprocess import Hist
def test_Hist_outputs():
    output_map = dict(out_file=dict(extensions=None), out_show=dict(extensions=None))
    outputs = Hist.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value