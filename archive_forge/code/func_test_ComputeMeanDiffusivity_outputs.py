from ..dti import ComputeMeanDiffusivity
def test_ComputeMeanDiffusivity_outputs():
    output_map = dict(md=dict(extensions=None))
    outputs = ComputeMeanDiffusivity.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value