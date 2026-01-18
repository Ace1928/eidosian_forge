from ..petstandarduptakevaluecomputation import PETStandardUptakeValueComputation
def test_PETStandardUptakeValueComputation_outputs():
    output_map = dict(csvFile=dict(extensions=None))
    outputs = PETStandardUptakeValueComputation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value