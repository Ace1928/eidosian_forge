from ..model import Binarize
def test_Binarize_outputs():
    output_map = dict(binary_file=dict(extensions=None), count_file=dict(extensions=None))
    outputs = Binarize.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value