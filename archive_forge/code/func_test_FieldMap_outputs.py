from ..preprocess import FieldMap
def test_FieldMap_outputs():
    output_map = dict(vdm=dict(extensions=None))
    outputs = FieldMap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value