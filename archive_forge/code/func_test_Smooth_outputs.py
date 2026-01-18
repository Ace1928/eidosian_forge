from ..preprocess import Smooth
def test_Smooth_outputs():
    output_map = dict(smoothed_file=dict(extensions=None))
    outputs = Smooth.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value