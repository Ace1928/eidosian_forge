from ..converters import BSplineToDeformationField
def test_BSplineToDeformationField_outputs():
    output_map = dict(defImage=dict(extensions=None))
    outputs = BSplineToDeformationField.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value