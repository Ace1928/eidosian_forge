from ..tracking import Tractography
def test_Tractography_outputs():
    output_map = dict(out_file=dict(extensions=None), out_seeds=dict(extensions=None))
    outputs = Tractography.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value