from ..dti import ComputeFractionalAnisotropy
def test_ComputeFractionalAnisotropy_outputs():
    output_map = dict(fa=dict(extensions=None))
    outputs = ComputeFractionalAnisotropy.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value