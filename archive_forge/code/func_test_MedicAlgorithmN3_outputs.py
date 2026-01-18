from ..developer import MedicAlgorithmN3
def test_MedicAlgorithmN3_outputs():
    output_map = dict(outInhomogeneity=dict(extensions=None), outInhomogeneity2=dict(extensions=None))
    outputs = MedicAlgorithmN3.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value