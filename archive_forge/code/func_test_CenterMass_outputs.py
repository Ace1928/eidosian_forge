from ..utils import CenterMass
def test_CenterMass_outputs():
    output_map = dict(cm=dict(), cm_file=dict(extensions=None), out_file=dict(extensions=None))
    outputs = CenterMass.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value